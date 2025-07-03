#!/usr/bin/env python3
"""
Ray Model Evaluator - Multi Node Multi GPU Version with Categorized Prompts
Script untuk evaluasi model yang sudah di-deploy di Ray cluster dengan dukungan multi node multi GPU
dan kategorisasi prompt berdasarkan panjang output dan jenis tugas
"""

import time
import json
import psutil
import GPUtil
import numpy as np
import logging
import argparse
import asyncio
from typing import List, Dict, Optional

import ray
from ray import serve

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

@ray.remote
class GPUMonitor:
    """Remote actor untuk monitoring GPU di setiap node"""
    
    def __init__(self):
        self.node_id = ray.get_runtime_context().get_node_id()
        self.node_ip = ray.util.get_node_ip_address()
        
    def get_node_info(self):
        """Get basic node information"""
        return {
            'node_id': self.node_id,
            'node_ip': self.node_ip,
            'hostname': ray._private.services.get_node_ip_address()
        }
    
    def get_gpu_info(self):
        """Get GPU information for this node"""
        try:
            gpus = GPUtil.getGPUs()
            gpu_info = []
            
            for i, gpu in enumerate(gpus):
                gpu_data = {
                    'node_id': self.node_id,
                    'node_ip': self.node_ip,
                    'gpu_id': gpu.id,
                    'gpu_name': gpu.name,
                    'total_memory': gpu.memoryTotal,
                    'driver': gpu.driver,
                    'uuid': gpu.uuid
                }
                gpu_info.append(gpu_data)
            
            return gpu_info
        except Exception as e:
            logger.warning(f"Could not get GPU info on node {self.node_id}: {e}")
            return []
    
    def collect_gpu_metrics(self):
        """Collect current GPU metrics for this node"""
        try:
            gpus = GPUtil.getGPUs()
            gpu_metrics = []
            
            for i, gpu in enumerate(gpus):
                metrics = {
                    'node_id': self.node_id,
                    'node_ip': self.node_ip,
                    'gpu_id': gpu.id,
                    'gpu_name': gpu.name,
                    'gpu_load': gpu.load * 100,
                    'gpu_memory_used': gpu.memoryUsed,
                    'gpu_memory_total': gpu.memoryTotal,
                    'gpu_memory_percent': (gpu.memoryUsed / gpu.memoryTotal * 100) if gpu.memoryTotal > 0 else 0,
                    'gpu_temperature': gpu.temperature if hasattr(gpu, 'temperature') else None,
                    'timestamp': time.time()
                }
                gpu_metrics.append(metrics)
            
            return gpu_metrics
        except Exception as e:
            logger.warning(f"Error collecting GPU metrics on node {self.node_id}: {e}")
            return []
    
    def get_system_metrics(self):
        """Get system metrics for this node"""
        try:
            return {
                'node_id': self.node_id,
                'node_ip': self.node_ip,
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'timestamp': time.time()
            }
        except Exception as e:
            logger.warning(f"Error collecting system metrics on node {self.node_id}: {e}")
            return {}

class PromptCategories:
    """Kategorisasi prompt untuk evaluasi yang lebih terstruktur"""
    
    SHORT_TEXT_PROMPTS = [
        "What is AI?",
        "Define machine learning.",
        "What is 2+2?",
        "Name three colors.",
        "What day is today?",
        "Define quantum computing.",
        "What is Python programming?",
        "Name a popular database.",
        "Define blockchain.",
        "What is cloud computing?"
    ]
    
    LONG_TEXT_PROMPTS = [
        "Write a detailed explanation of how artificial intelligence works, including its history, current applications, and future prospects.",
        "Explain the complete process of machine learning from data collection to model deployment, with examples.",
        "Write a comprehensive guide on renewable energy sources, their advantages, disadvantages, and implementation challenges.",
        "Describe the evolution of computer technology from the 1940s to present day, including major milestones.",
        "Write a detailed analysis of climate change causes, effects, and potential solutions.",
        "Explain how the internet works, from physical infrastructure to protocols and data transmission.",
        "Write a comprehensive overview of modern programming languages and their use cases.",
        "Describe the complete software development lifecycle with best practices and methodologies."
    ]
    
    CREATIVE_PROMPTS = [
        "Write a short story about a robot learning to paint.",
        "Create a poem about the beauty of mathematics.",
        "Write a creative dialogue between two AI systems.",
        "Compose a song about space exploration.",
        "Write a fictional news report from the year 2050.",
        "Create a humorous conversation between a programmer and their code.",
        "Write a short play about time travel.",
        "Compose a creative essay about the color blue."
    ]
    
    TECHNICAL_PROMPTS = [
        "Explain the differences between SQL and NoSQL databases with examples.",
        "Describe the REST API architectural style and its principles.",
        "Explain how containerization works with Docker and Kubernetes.",
        "Describe the MVC (Model-View-Controller) design pattern.",
        "Explain the concept of microservices architecture.",
        "Describe how version control systems like Git work.",
        "Explain the principles of object-oriented programming.",
        "Describe the difference between relational and non-relational databases."
    ]
    
    REASONING_PROMPTS = [
        "If you have 100 books and you read 3 books per week, how many weeks will it take to read all books? Show your reasoning.",
        "A train leaves Station A at 2 PM traveling at 60 mph. Another train leaves Station B at 3 PM traveling at 80 mph toward Station A. If the stations are 200 miles apart, when will the trains meet?",
        "You have a 3-gallon jug and a 5-gallon jug. How can you measure exactly 4 gallons of water?",
        "If all roses are flowers, and some flowers are red, can we conclude that some roses are red? Explain your reasoning.",
        "In a group of 30 people, everyone shakes hands exactly once with every other person. How many handshakes occur in total?",
        "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
        "You have 12 balls, one of which is heavier than the others. Using a balance scale only 3 times, how can you identify the heavy ball?",
        "If a shirt costs $20 after a 20% discount, what was the original price?"
    ]
    
    @classmethod
    def get_all_categories(cls):
        """Return all available prompt categories"""
        return {
            'short_text': cls.SHORT_TEXT_PROMPTS,
            'long_text': cls.LONG_TEXT_PROMPTS,
            'creative': cls.CREATIVE_PROMPTS,
            'technical': cls.TECHNICAL_PROMPTS,
            'reasoning': cls.REASONING_PROMPTS
        }
    
    @classmethod
    def get_mixed_prompts(cls, count_per_category=2):
        """Get a mixed set of prompts from all categories"""
        categories = cls.get_all_categories()
        mixed_prompts = []
        
        for category_name, prompts in categories.items():
            selected = prompts[:count_per_category]
            for prompt in selected:
                mixed_prompts.append({
                    'prompt': prompt,
                    'category': category_name,
                    'expected_length': 'short' if category_name == 'short_text' else 'medium_to_long'
                })
        
        return mixed_prompts

class MultiNodeMultiGPUModelEvaluator:
    """Evaluator untuk model yang sudah di-deploy dengan dukungan multi node multi GPU"""

    def __init__(self, ray_address=None):
        self.ray_address = ray_address
        self.connected = False
        self.deployment_available = False
        self.gpu_monitors = []
        self.cluster_gpu_info = []
        self.total_gpu_count = 0
        self.node_count = 0

    async def initialize_gpu_monitors(self):
        """Initialize GPU monitors on all nodes in the cluster"""
        logger.info("Initializing GPU monitors across all nodes...")
        
        try:
            # Get all nodes in the cluster
            nodes = ray.nodes()
            active_nodes = [node for node in nodes if node['Alive']]
            self.node_count = len(active_nodes)
            
            logger.info(f"Found {self.node_count} active nodes in cluster")
            
            # Create GPU monitor on each node
            for i, node in enumerate(active_nodes):
                node_id = node['NodeID']
                node_ip = node['NodeManagerAddress']
                
                try:
                    # Create GPU monitor actor on specific node
                    monitor = GPUMonitor.options(
                        resources={f"node:{node_id}": 0.01}
                    ).remote()
                    
                    self.gpu_monitors.append({
                        'monitor': monitor,
                        'node_id': node_id,
                        'node_ip': node_ip
                    })
                    
                    logger.info(f"Created GPU monitor on node {i+1}: {node_ip}")
                    
                except Exception as e:
                    logger.warning(f"Failed to create GPU monitor on node {node_ip}: {e}")
            
            # Collect GPU information from all nodes
            await self.collect_cluster_gpu_info()
            
            logger.info(f"Initialized {len(self.gpu_monitors)} GPU monitors")
            logger.info(f"Total GPUs in cluster: {self.total_gpu_count}")
            
        except Exception as e:
            logger.error(f"Failed to initialize GPU monitors: {e}")

    async def collect_cluster_gpu_info(self):
        """Collect GPU information from all nodes"""
        self.cluster_gpu_info = []
        self.total_gpu_count = 0
        
        if not self.gpu_monitors:
            logger.warning("No GPU monitors available")
            return
        
        # Collect GPU info from all monitors
        gpu_info_futures = []
        for monitor_info in self.gpu_monitors:
            future = monitor_info['monitor'].get_gpu_info.remote()
            gpu_info_futures.append(future)
        
        try:
            gpu_info_results = await asyncio.gather(*[
                asyncio.create_task(asyncio.wrap_future(ray.get_async(future)))
                for future in gpu_info_futures
            ])
            
            # Combine results from all nodes
            for node_gpu_info in gpu_info_results:
                if node_gpu_info:
                    self.cluster_gpu_info.extend(node_gpu_info)
                    self.total_gpu_count += len(node_gpu_info)
            
            # Log GPU information
            if self.cluster_gpu_info:
                logger.info("Cluster GPU Information:")
                nodes_gpu_count = {}
                for gpu in self.cluster_gpu_info:
                    node_ip = gpu['node_ip']
                    if node_ip not in nodes_gpu_count:
                        nodes_gpu_count[node_ip] = []
                    nodes_gpu_count[node_ip].append(gpu)
                
                for node_ip, gpus in nodes_gpu_count.items():
                    logger.info(f"  Node {node_ip}: {len(gpus)} GPU(s)")
                    for gpu in gpus:
                        logger.info(f"    GPU {gpu['gpu_id']}: {gpu['gpu_name']} ({gpu['total_memory']}MB)")
            
        except Exception as e:
            logger.error(f"Failed to collect cluster GPU info: {e}")

    async def collect_cluster_gpu_metrics(self):
        """Collect GPU metrics from all nodes"""
        if not self.gpu_monitors:
            return []
        
        # Collect metrics from all monitors
        metrics_futures = []
        for monitor_info in self.gpu_monitors:
            future = monitor_info['monitor'].collect_gpu_metrics.remote()
            metrics_futures.append(future)
        
        try:
            metrics_results = await asyncio.gather(*[
                asyncio.create_task(asyncio.wrap_future(ray.get_async(future)))
                for future in metrics_futures
            ])
            
            # Combine results from all nodes
            all_gpu_metrics = []
            for node_metrics in metrics_results:
                if node_metrics:
                    all_gpu_metrics.extend(node_metrics)
            
            return all_gpu_metrics
            
        except Exception as e:
            logger.warning(f"Error collecting cluster GPU metrics: {e}")
            return []

    async def collect_cluster_system_metrics(self):
        """Collect system metrics from all nodes"""
        if not self.gpu_monitors:
            return []
        
        # Collect system metrics from all monitors
        system_futures = []
        for monitor_info in self.gpu_monitors:
            future = monitor_info['monitor'].get_system_metrics.remote()
            system_futures.append(future)
        
        try:
            system_results = await asyncio.gather(*[
                asyncio.create_task(asyncio.wrap_future(ray.get_async(future)))
                for future in system_futures
            ])
            
            # Filter out empty results
            all_system_metrics = [metrics for metrics in system_results if metrics]
            return all_system_metrics
            
        except Exception as e:
            logger.warning(f"Error collecting cluster system metrics: {e}")
            return []

    def calculate_cluster_gpu_averages(self, gpu_metrics_list: List[List[Dict]]):
        """Calculate average metrics across all GPUs in the cluster"""
        if not gpu_metrics_list:
            return []
        
        # Flatten all GPU metrics
        all_gpu_data = {}
        
        for measurement in gpu_metrics_list:
            for gpu_metric in measurement:
                # Create unique key for each GPU (node_id + gpu_id)
                gpu_key = f"{gpu_metric['node_id']}-{gpu_metric['gpu_id']}"
                
                if gpu_key not in all_gpu_data:
                    all_gpu_data[gpu_key] = {
                        'node_id': gpu_metric['node_id'],
                        'node_ip': gpu_metric['node_ip'],
                        'gpu_id': gpu_metric['gpu_id'],
                        'gpu_name': gpu_metric['gpu_name'],
                        'gpu_memory_total': gpu_metric['gpu_memory_total'],
                        'loads': [],
                        'memory_percents': [],
                        'memory_used': [],
                        'temperatures': []
                    }
                
                all_gpu_data[gpu_key]['loads'].append(gpu_metric['gpu_load'])
                all_gpu_data[gpu_key]['memory_percents'].append(gpu_metric['gpu_memory_percent'])
                all_gpu_data[gpu_key]['memory_used'].append(gpu_metric['gpu_memory_used'])
                
                if gpu_metric['gpu_temperature'] is not None:
                    all_gpu_data[gpu_key]['temperatures'].append(gpu_metric['gpu_temperature'])
        
        # Calculate averages for each GPU
        averaged_gpus = []
        for gpu_key, gpu_data in all_gpu_data.items():
            avg_gpu = {
                'node_id': gpu_data['node_id'],
                'node_ip': gpu_data['node_ip'],
                'gpu_id': gpu_data['gpu_id'],
                'gpu_name': gpu_data['gpu_name'],
                'gpu_memory_total': gpu_data['gpu_memory_total'],
                'avg_gpu_load': np.mean(gpu_data['loads']) if gpu_data['loads'] else 0,
                'avg_gpu_memory_percent': np.mean(gpu_data['memory_percents']) if gpu_data['memory_percents'] else 0,
                'avg_gpu_memory_used': np.mean(gpu_data['memory_used']) if gpu_data['memory_used'] else 0,
                'avg_gpu_temperature': np.mean(gpu_data['temperatures']) if gpu_data['temperatures'] else None,
                'std_gpu_load': np.std(gpu_data['loads']) if gpu_data['loads'] else 0,
                'std_gpu_memory_percent': np.std(gpu_data['memory_percents']) if gpu_data['memory_percents'] else 0,
                'measurements_count': len(gpu_data['loads'])
            }
            averaged_gpus.append(avg_gpu)
        
        # Sort by node_ip and gpu_id for consistent ordering
        averaged_gpus.sort(key=lambda x: (x['node_ip'], x['gpu_id']))
        
        return averaged_gpus

    def connect_to_cluster(self):
        """Connect ke Ray cluster"""
        logger.info(f"Connecting to Ray cluster: {self.ray_address or 'auto-detect'}")

        try:
            if self.ray_address:
                ray.init(address=self.ray_address, ignore_reinit_error=True)
            else:
                ray.init(address='auto', ignore_reinit_error=True)

            self.connected = True
            logger.info("Connected to Ray cluster")
            
            return True

        except Exception as e:
            logger.error(f"Failed to connect to cluster: {e}")
            return False

    def check_deployment(self):
        """Check if model deployment is available"""
        if not self.connected:
            logger.error("Not connected to Ray cluster")
            return False

        try:
            # Get status information as a ServeStatus object
            status_info = serve.status()
            logger.info(f"Deployment info: {status_info}")

            # Access the applications dictionary from the ServeStatus object
            if hasattr(status_info, "applications") and "vllm-model" in status_info.applications:
                self.deployment_available = True
                app_status = status_info.applications["vllm-model"].status.value
                logger.info(f"Deployment status: {app_status}")
                return True
            else:
                logger.info("No deployment found")
                return False
        except Exception as e:
            logger.error(f"Error checking deployment: {e}")
            return False

    def analyze_response_characteristics(self, response_text: str):
        """Analyze characteristics of the generated response"""
        if not response_text:
            return {}
        
        words = response_text.split()
        sentences = response_text.split('.')
        paragraphs = response_text.split('\n\n')
        
        return {
            'word_count': len(words),
            'sentence_count': len([s for s in sentences if s.strip()]),
            'paragraph_count': len([p for p in paragraphs if p.strip()]),
            'avg_words_per_sentence': len(words) / max(1, len([s for s in sentences if s.strip()])),
            'character_count': len(response_text),
            'response_length_category': self.categorize_response_length(len(words))
        }
    
    def categorize_response_length(self, word_count: int):
        """Categorize response based on word count"""
        if word_count < 20:
            return 'very_short'
        elif word_count < 50:
            return 'short'
        elif word_count < 150:
            return 'medium'
        elif word_count < 300:
            return 'long'
        else:
            return 'very_long'

    async def test_inference(self, prompt: str, max_tokens: int = 100, temperature: float = 0.7, category: str = None):
        """Test single inference with detailed multi-node multi-GPU monitoring and response analysis"""
        if not self.deployment_available:
            logger.error("No deployment available")
            return None

        logger.info(f"Testing inference ({category or 'unknown'}): {prompt[:40]}...")

        # Collect cluster metrics before
        gpu_metrics_before = await self.collect_cluster_gpu_metrics()
        system_metrics_before = await self.collect_cluster_system_metrics()

        # Time the inference
        start_time = time.time()

        try:
            # Get deployment handle
            handle = serve.get_deployment_handle("vllm-model", "vllm-model")

            # Make inference request
            result = await handle.generate.remote(prompt, max_tokens, temperature)

            end_time = time.time()
            total_time = end_time - start_time

            # Collect cluster metrics after
            gpu_metrics_after = await self.collect_cluster_gpu_metrics()
            system_metrics_after = await self.collect_cluster_system_metrics()

            if result and "error" not in result:
                throughput = result['completion_tokens'] / total_time if total_time > 0 else 0

                # Analyze response characteristics
                response_analysis = self.analyze_response_characteristics(result['text'])

                # Calculate averages for backward compatibility
                avg_gpu_load = 0
                avg_gpu_memory = 0
                avg_cpu = 0
                avg_ram = 0
                
                if gpu_metrics_before and gpu_metrics_after:
                    all_gpu_metrics = gpu_metrics_before + gpu_metrics_after
                    avg_gpu_load = np.mean([gpu['gpu_load'] for gpu in all_gpu_metrics])
                    avg_gpu_memory = np.mean([gpu['gpu_memory_percent'] for gpu in all_gpu_metrics])
                
                if system_metrics_before and system_metrics_after:
                    all_system_metrics = system_metrics_before + system_metrics_after
                    avg_cpu = np.mean([sys['cpu_percent'] for sys in all_system_metrics])
                    avg_ram = np.mean([sys['memory_percent'] for sys in all_system_metrics])

                evaluation_result = {
                    "prompt": prompt,
                    "prompt_category": category,
                    "response": result['text'],
                    "response_analysis": response_analysis,
                    "model": result.get('model', 'unknown'),
                    "total_time": total_time,
                    "prompt_tokens": result['prompt_tokens'],
                    "completion_tokens": result['completion_tokens'],
                    "total_tokens": result['total_tokens'],
                    "throughput": throughput,
                    "tokens_per_second": throughput,
                    "cluster_cpu_usage": avg_cpu,
                    "cluster_ram_usage": avg_ram,
                    "cluster_gpu_usage": avg_gpu_load,
                    "cluster_gpu_memory": avg_gpu_memory,
                    "timestamp": time.time(),
                    # New detailed cluster metrics
                    "cluster_gpu_metrics_before": gpu_metrics_before,
                    "cluster_gpu_metrics_after": gpu_metrics_after,
                    "cluster_system_metrics_before": system_metrics_before,
                    "cluster_system_metrics_after": system_metrics_after,
                    "total_gpu_count": len(gpu_metrics_before) if gpu_metrics_before else 0,
                    "node_count": self.node_count
                }

                logger.info(f"✓ Success: {total_time:.2f}s, {throughput:.1f} tokens/s, "
                          f"{response_analysis['word_count']} words ({response_analysis['response_length_category']})")
                
                if gpu_metrics_after:
                    # Group by node for logging
                    nodes_gpu = {}
                    for gpu in gpu_metrics_after:
                        node_ip = gpu['node_ip']
                        if node_ip not in nodes_gpu:
                            nodes_gpu[node_ip] = []
                        nodes_gpu[node_ip].append(gpu)
                    
                    gpu_summary = []
                    for node_ip, gpus in nodes_gpu.items():
                        node_summary = f"{node_ip}:[" + ",".join([f"GPU{gpu['gpu_id']}={gpu['gpu_load']:.1f}%" for gpu in gpus]) + "]"
                        gpu_summary.append(node_summary)
                    
                    logger.info(f"  Cluster GPU utilization: {' '.join(gpu_summary)}")
                
                return evaluation_result
            else:
                logger.error(f"Inference failed: {result}")
                return None

        except Exception as e:
            logger.error(f"Inference error: {e}")
            return None

    async def run_single_prompt_evaluation(self, prompt_data, iterations: int = 3,
                                          max_tokens: int = 100, temperature: float = 0.7):
        """Run multiple iterations untuk single prompt dengan detailed cluster GPU tracking"""
        
        # Handle both string and dict prompt formats
        if isinstance(prompt_data, str):
            prompt = prompt_data
            category = "uncategorized"
        else:
            prompt = prompt_data.get('prompt', prompt_data)
            category = prompt_data.get('category', 'uncategorized')
        
        logger.info(f"Evaluating {category} prompt with {iterations} iterations: {prompt[:50]}...")

        results = []
        all_cluster_gpu_metrics_before = []
        all_cluster_gpu_metrics_after = []

        for i in range(iterations):
            result = await self.test_inference(prompt, max_tokens, temperature, category)
            if result:
                results.append(result)
                
                # Collect cluster GPU metrics for averaging
                if result.get('cluster_gpu_metrics_before'):
                    all_cluster_gpu_metrics_before.append(result['cluster_gpu_metrics_before'])
                if result.get('cluster_gpu_metrics_after'):
                    all_cluster_gpu_metrics_after.append(result['cluster_gpu_metrics_after'])
                
                logger.info(f"  Iteration {i+1}: {result['total_time']:.2f}s, "
                            f"{result['throughput']:.1f} tok/s, "
                            f"{result['response_analysis']['word_count']} words")

            # Small delay between iterations
            if i < iterations - 1:
                await asyncio.sleep(0.5)

        if results:
            # Calculate cluster GPU averages across all iterations
            combined_cluster_gpu_metrics = all_cluster_gpu_metrics_before + all_cluster_gpu_metrics_after
            averaged_cluster_gpu_metrics = self.calculate_cluster_gpu_averages(combined_cluster_gpu_metrics)
            
            # Calculate response analysis averages
            avg_word_count = np.mean([r['response_analysis']['word_count'] for r in results])
            avg_sentence_count = np.mean([r['response_analysis']['sentence_count'] for r in results])
            avg_character_count = np.mean([r['response_analysis']['character_count'] for r in results])
            
            # Calculate overall averages
            avg_result = {
                "prompt": prompt,
                "prompt_category": category,
                "iterations": len(results),
                "avg_total_time": np.mean([r['total_time'] for r in results]),
                "avg_throughput": np.mean([r['throughput'] for r in results]),
                "avg_prompt_tokens": np.mean([r['prompt_tokens'] for r in results]),
                "avg_completion_tokens": np.mean([r['completion_tokens'] for r in results]),
                "avg_cluster_cpu": np.mean([r['cluster_cpu_usage'] for r in results]),
                "avg_cluster_ram": np.mean([r['cluster_ram_usage'] for r in results]),
                "avg_cluster_gpu": np.mean([r['cluster_gpu_usage'] for r in results]),
                "avg_cluster_gpu_mem": np.mean([r['cluster_gpu_memory'] for r in results]),
                "std_total_time": np.std([r['total_time'] for r in results]),
                "std_throughput": np.std([r['throughput'] for r in results]),
                # Response analysis averages
                "avg_response_analysis": {
                    "avg_word_count": avg_word_count,
                    "avg_sentence_count": avg_sentence_count,
                    "avg_character_count": avg_character_count,
                    "avg_words_per_sentence": np.mean([r['response_analysis']['avg_words_per_sentence'] for r in results]),
                    "dominant_length_category": max(set([r['response_analysis']['response_length_category'] for r in results]), 
                                                  key=[r['response_analysis']['response_length_category'] for r in results].count)
                },
                "detailed_results": results,
                "model": results[0].get('model', 'unknown'),
                # New detailed cluster metrics
                "total_gpu_count": results[0].get('total_gpu_count', 0),
                "node_count": results[0].get('node_count', 0),
                "cluster_gpu_metrics": averaged_cluster_gpu_metrics,
                "cluster_gpu_info": self.cluster_gpu_info
            }

            logger.info(f"Average results: {avg_result['avg_total_time']:.2f}s, "
                        f"{avg_result['avg_throughput']:.1f} tok/s, "
                        f"{avg_word_count:.0f} words")
            
            # Log cluster GPU averages
            if averaged_cluster_gpu_metrics:
                # Group by node
                nodes_gpu = {}
                for gpu in averaged_cluster_gpu_metrics:
                    node_ip = gpu['node_ip']
                    if node_ip not in nodes_gpu:
                        nodes_gpu[node_ip] = []
                    nodes_gpu[node_ip].append(gpu)
                
                node_summaries = []
                for node_ip, gpus in nodes_gpu.items():
                    gpu_summary = ", ".join([f"GPU{gpu['gpu_id']}: {gpu['avg_gpu_load']:.1f}% load, {gpu['avg_gpu_memory_percent']:.1f}% mem" 
                                           for gpu in gpus])
                    node_summaries.append(f"{node_ip}: [{gpu_summary}]")
                
                logger.info(f"Cluster GPU averages: {' | '.join(node_summaries)}")
            
            return avg_result

        return None

    async def run_batch_evaluation(self, prompts: List, iterations: int = 3,
                                  max_tokens: int = 100, temperature: float = 0.7):
        """Run evaluation untuk multiple prompts"""
        logger.info(f"Running batch evaluation: {len(prompts)} prompts x {iterations} iterations")
        logger.info(f"Cluster: {self.node_count} nodes, {self.total_gpu_count} GPUs")

        batch_results = []
        start_time = time.time()

        for i, prompt_data in enumerate(prompts):
            logger.info(f"\n--- Prompt {i+1}/{len(prompts)} ---")
            
            result = await self.run_single_prompt_evaluation(
                prompt_data, iterations, max_tokens, temperature
            )
            
            if result:
                batch_results.append(result)
            
            # Progress update
            elapsed = time.time() - start_time
            avg_time_per_prompt = elapsed / (i + 1)
            estimated_remaining = avg_time_per_prompt * (len(prompts) - i - 1)
            
            logger.info(f"Progress: {i+1}/{len(prompts)} completed, "
                       f"estimated remaining: {estimated_remaining:.1f}s")

        total_time = time.time() - start_time

        if batch_results:
            # Calculate overall statistics
            batch_summary = self.calculate_batch_summary(batch_results, total_time)
            logger.info(f"\n=== Batch Evaluation Complete ===")
            logger.info(f"Total time: {total_time:.2f}s")
            logger.info(f"Average throughput: {batch_summary['overall_avg_throughput']:.2f} tokens/s")
            logger.info(f"Success rate: {batch_summary['success_rate']:.1f}%")
            
            return batch_summary

        return None

    def calculate_batch_summary(self, batch_results: List[Dict], total_time: float):
        """Calculate comprehensive batch evaluation summary"""
        if not batch_results:
            return {}

        # Overall performance metrics
        all_throughputs = [r['avg_throughput'] for r in batch_results]
        all_times = [r['avg_total_time'] for r in batch_results]
        all_prompt_tokens = [r['avg_prompt_tokens'] for r in batch_results]
        all_completion_tokens = [r['avg_completion_tokens'] for r in batch_results]
        all_word_counts = [r['avg_response_analysis']['avg_word_count'] for r in batch_results if 'avg_response_analysis' in r]

        # Resource utilization metrics
        all_gpu_usage = [r['avg_cluster_gpu'] for r in batch_results]
        all_gpu_memory = [r['avg_cluster_gpu_mem'] for r in batch_results]
        all_cpu_usage = [r['avg_cluster_cpu'] for r in batch_results]
        all_ram_usage = [r['avg_cluster_ram'] for r in batch_results]

        # Category-wise analysis
        category_stats = {}
        for result in batch_results:
            category = result.get('prompt_category', 'unknown')
            if category not in category_stats:
                category_stats[category] = {
                    'count': 0,
                    'throughputs': [],
                    'times': [],
                    'word_counts': []
                }
            
            category_stats[category]['count'] += 1
            category_stats[category]['throughputs'].append(result['avg_throughput'])
            category_stats[category]['times'].append(result['avg_total_time'])
            if 'avg_response_analysis' in result:
                category_stats[category]['word_counts'].append(result['avg_response_analysis']['avg_word_count'])

        # Calculate category averages
        for category, stats in category_stats.items():
            stats['avg_throughput'] = np.mean(stats['throughputs'])
            stats['avg_time'] = np.mean(stats['times'])
            stats['avg_word_count'] = np.mean(stats['word_counts']) if stats['word_counts'] else 0

        # Cluster resource summary
        cluster_summary = {
            'total_nodes': batch_results[0].get('node_count', 0),
            'total_gpus': batch_results[0].get('total_gpu_count', 0),
            'avg_cluster_gpu_utilization': np.mean(all_gpu_usage),
            'avg_cluster_gpu_memory': np.mean(all_gpu_memory),
            'avg_cluster_cpu': np.mean(all_cpu_usage),
            'avg_cluster_ram': np.mean(all_ram_usage)
        }

        # Performance percentiles
        throughput_percentiles = {
            'p50': np.percentile(all_throughputs, 50),
            'p95': np.percentile(all_throughputs, 95),
            'p99': np.percentile(all_throughputs, 99)
        }

        batch_summary = {
            'total_prompts': len(batch_results),
            'total_evaluation_time': total_time,
            'success_rate': (len(batch_results) / len(batch_results)) * 100 if batch_results else 0,
            
            # Overall performance
            'overall_avg_throughput': np.mean(all_throughputs),
            'overall_std_throughput': np.std(all_throughputs),
            'overall_avg_time': np.mean(all_times),
            'overall_std_time': np.std(all_times),
            
            # Token statistics
            'avg_prompt_tokens': np.mean(all_prompt_tokens),
            'avg_completion_tokens': np.mean(all_completion_tokens),
            'total_tokens_processed': sum(all_prompt_tokens) + sum(all_completion_tokens),
            
            # Response characteristics
            'avg_response_word_count': np.mean(all_word_counts) if all_word_counts else 0,
            
            # Performance percentiles
            'throughput_percentiles': throughput_percentiles,
            
            # Category analysis
            'category_analysis': category_stats,
            
            # Cluster resource utilization
            'cluster_resource_summary': cluster_summary,
            
            # Detailed results
            'detailed_results': batch_results,
            
            'timestamp': time.time()
        }

        return batch_summary

    def save_results(self, results: Dict, filename: str = None):
        """Save evaluation results to JSON file"""
        if not filename:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"ray_model_evaluation_{timestamp}.json"

        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Results saved to: {filename}")
            return filename
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            return None

    def print_summary_report(self, results: Dict):
        """Print a formatted summary report"""
        if not results:
            logger.info("No results to display")
            return

        print("\n" + "="*80)
        print("RAY MODEL EVALUATION SUMMARY REPORT")
        print("="*80)
        
        # Cluster information
        cluster_info = results.get('cluster_resource_summary', {})
        print(f"\nCluster Configuration:")
        print(f"  Nodes: {cluster_info.get('total_nodes', 'N/A')}")
        print(f"  Total GPUs: {cluster_info.get('total_gpus', 'N/A')}")
        
        # Overall performance
        print(f"\nOverall Performance:")
        print(f"  Total Prompts: {results.get('total_prompts', 0)}")
        print(f"  Success Rate: {results.get('success_rate', 0):.1f}%")
        print(f"  Average Throughput: {results.get('overall_avg_throughput', 0):.2f} tokens/s")
        print(f"  Average Response Time: {results.get('overall_avg_time', 0):.2f}s")
        print(f"  Total Evaluation Time: {results.get('total_evaluation_time', 0):.2f}s")
        
        # Throughput percentiles
        percentiles = results.get('throughput_percentiles', {})
        if percentiles:
            print(f"\nThroughput Percentiles:")
            print(f"  P50: {percentiles.get('p50', 0):.2f} tokens/s")
            print(f"  P95: {percentiles.get('p95', 0):.2f} tokens/s")
            print(f"  P99: {percentiles.get('p99', 0):.2f} tokens/s")
        
        # Category analysis
        category_analysis = results.get('category_analysis', {})
        if category_analysis:
            print(f"\nCategory Analysis:")
            for category, stats in category_analysis.items():
                print(f"  {category.upper()}:")
                print(f"    Count: {stats.get('count', 0)}")
                print(f"    Avg Throughput: {stats.get('avg_throughput', 0):.2f} tokens/s")
                print(f"    Avg Time: {stats.get('avg_time', 0):.2f}s")
                print(f"    Avg Word Count: {stats.get('avg_word_count', 0):.0f}")
        
        # Resource utilization
        print(f"\nCluster Resource Utilization:")
        print(f"  GPU Utilization: {cluster_info.get('avg_cluster_gpu_utilization', 0):.1f}%")
        print(f"  GPU Memory: {cluster_info.get('avg_cluster_gpu_memory', 0):.1f}%")
        print(f"  CPU Usage: {cluster_info.get('avg_cluster_cpu', 0):.1f}%")
        print(f"  RAM Usage: {cluster_info.get('avg_cluster_ram', 0):.1f}%")
        
        print("="*80)

    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up resources...")
        
        # Cleanup GPU monitors
        for monitor_info in self.gpu_monitors:
            try:
                ray.kill(monitor_info['monitor'])
            except:
                pass
        
        self.gpu_monitors.clear()
        logger.info("Cleanup completed")

async def main():
    """Main function untuk menjalankan evaluasi"""
    parser = argparse.ArgumentParser(description='Ray Model Evaluator with Multi-Node Multi-GPU Support')
    parser.add_argument('--ray-address', type=str, help='Ray cluster address (default: auto-detect)')
    parser.add_argument('--iterations', type=int, default=3, help='Number of iterations per prompt')
    parser.add_argument('--max-tokens', type=int, default=100, help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.7, help='Generation temperature')
    parser.add_argument('--category', type=str, choices=['short_text', 'long_text', 'creative', 'technical', 'reasoning', 'mixed'], 
                       default='mixed', help='Prompt category to evaluate')
    parser.add_argument('--prompts-per-category', type=int, default=2, help='Number of prompts per category for mixed evaluation')
    parser.add_argument('--output-file', type=str, help='Output file for results (default: auto-generated)')
    
    args = parser.parse_args()

    # Initialize evaluator
    evaluator = MultiNodeMultiGPUModelEvaluator(ray_address=args.ray_address)

    try:
        # Connect to cluster
        if not evaluator.connect_to_cluster():
            logger.error("Failed to connect to Ray cluster")
            return

        # Initialize GPU monitors
        await evaluator.initialize_gpu_monitors()

        # Check deployment
        if not evaluator.check_deployment():
            logger.error("Model deployment not available")
            return

        # Select prompts based on category
        if args.category == 'mixed':
            prompts = PromptCategories.get_mixed_prompts(args.prompts_per_category)
        else:
            categories = PromptCategories.get_all_categories()
            if args.category in categories:
                category_prompts = categories[args.category]
                prompts = [{'prompt': p, 'category': args.category} for p in category_prompts[:args.prompts_per_category * 5]]
            else:
                logger.error(f"Unknown category: {args.category}")
                return

        logger.info(f"Selected {len(prompts)} prompts from category: {args.category}")

        # Run evaluation
        results = await evaluator.run_batch_evaluation(
            prompts=prompts,
            iterations=args.iterations,
            max_tokens=args.max_tokens,
            temperature=args.temperature
        )

        if results:
            # Print summary
            evaluator.print_summary_report(results)
            
            # Save results
            output_file = evaluator.save_results(results, args.output_file)
            if output_file:
                logger.info(f"Detailed results saved to: {output_file}")
        else:
            logger.error("Evaluation failed")

    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
    except Exception as e:
        logger.error(f"Evaluation error: {e}")
    finally:
        # Cleanup
        await evaluator.cleanup()
        
        # Disconnect from Ray
        try:
            ray.shutdown()
            logger.info("Disconnected from Ray cluster")
        except:
            pass

if __name__ == "__main__":
    asyncio.run(main())