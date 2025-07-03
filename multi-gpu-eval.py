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
            'hostname': psutil.uname().node if hasattr(psutil.uname(), 'node') else 'unknown'
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
            logger.warning(f"Node {self.node_id}: Could not get GPU info: {e}")
            return []
    
    def collect_gpu_metrics(self):
        """Collect current GPU metrics for this node"""
        try:
            gpus = GPUtil.getGPUs()
            metrics = []
            
            for i, gpu in enumerate(gpus):
                gpu_metrics = {
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
                metrics.append(gpu_metrics)
            
            return metrics
        except Exception as e:
            logger.warning(f"Node {self.node_id}: Error collecting GPU metrics: {e}")
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
            logger.warning(f"Node {self.node_id}: Error collecting system metrics: {e}")
            return {
                'node_id': self.node_id,
                'node_ip': self.node_ip,
                'cpu_percent': 0,
                'memory_percent': 0,
                'timestamp': time.time()
            }

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
        self.cluster_info = {}

    def setup_gpu_monitors(self):
        """Setup GPU monitors di semua node dalam cluster"""
        if not self.connected:
            logger.error("Not connected to Ray cluster")
            return False

        try:
            # Get all nodes in cluster
            nodes = ray.nodes()
            logger.info(f"Found {len(nodes)} nodes in cluster")
            
            self.gpu_monitors = []
            
            # Create GPU monitor untuk setiap node yang alive
            for node in nodes:
                if node.get('Alive', False):
                    node_id = node.get('NodeID')
                    node_ip = node.get('NodeManagerAddress', 'unknown')
                    
                    try:
                        # Create GPU monitor di node ini
                        monitor = GPUMonitor.options(
                            scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                                node_id=node_id,
                                soft=False
                            )
                        ).remote()
                        
                        # Test if monitor is working
                        node_info = ray.get(monitor.get_node_info.remote(), timeout=5)
                        logger.info(f"GPU monitor created for node {node_info['hostname']} ({node_info['node_ip']})")
                        
                        self.gpu_monitors.append(monitor)
                        
                    except Exception as e:
                        logger.warning(f"Failed to create GPU monitor for node {node_id}: {e}")
                        continue
            
            if self.gpu_monitors:
                logger.info(f"Successfully created {len(self.gpu_monitors)} GPU monitors")
                return True
            else:
                logger.warning("No GPU monitors could be created")
                return False
                
        except Exception as e:
            logger.error(f"Failed to setup GPU monitors: {e}")
            return False

    def get_cluster_gpu_info(self):
        """Get GPU information dari semua node dalam cluster"""
        if not self.gpu_monitors:
            logger.warning("No GPU monitors available")
            return False

        try:
            # Collect GPU info dari semua monitors
            gpu_info_futures = [monitor.get_gpu_info.remote() for monitor in self.gpu_monitors]
            gpu_info_results = ray.get(gpu_info_futures, timeout=10)
            
            self.cluster_gpu_info = []
            total_gpus = 0
            
            for gpu_info_list in gpu_info_results:
                if gpu_info_list:
                    self.cluster_gpu_info.extend(gpu_info_list)
                    total_gpus += len(gpu_info_list)
            
            # Group by node
            nodes_summary = {}
            for gpu_info in self.cluster_gpu_info:
                node_key = f"{gpu_info['node_ip']} ({gpu_info['node_id'][:8]})"
                if node_key not in nodes_summary:
                    nodes_summary[node_key] = []
                nodes_summary[node_key].append(gpu_info)
            
            logger.info(f"Cluster GPU Summary: {total_gpus} total GPUs across {len(nodes_summary)} nodes")
            for node_key, gpus in nodes_summary.items():
                gpu_names = [gpu['gpu_name'] for gpu in gpus]
                logger.info(f"  Node {node_key}: {len(gpus)} GPUs - {', '.join(gpu_names)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to get cluster GPU info: {e}")
            return False

    def collect_cluster_gpu_metrics(self):
        """Collect GPU metrics dari semua node"""
        if not self.gpu_monitors:
            return []

        try:
            # Collect metrics dari semua monitors
            metrics_futures = [monitor.collect_gpu_metrics.remote() for monitor in self.gpu_monitors]
            metrics_results = ray.get(metrics_futures, timeout=10)
            
            all_gpu_metrics = []
            for metrics_list in metrics_results:
                if metrics_list:
                    all_gpu_metrics.extend(metrics_list)
            
            return all_gpu_metrics
            
        except Exception as e:
            logger.warning(f"Error collecting cluster GPU metrics: {e}")
            return []

    def collect_cluster_system_metrics(self):
        """Collect system metrics dari semua node"""
        if not self.gpu_monitors:
            return []

        try:
            # Collect system metrics dari semua monitors
            system_futures = [monitor.get_system_metrics.remote() for monitor in self.gpu_monitors]
            system_results = ray.get(system_futures, timeout=10)
            
            return [result for result in system_results if result]
            
        except Exception as e:
            logger.warning(f"Error collecting cluster system metrics: {e}")
            return []

    def calculate_cluster_gpu_averages(self, gpu_metrics_list: List[List[Dict]]):
        """Calculate average GPU metrics across cluster"""
        if not gpu_metrics_list:
            return []
        
        # Flatten all measurements
        all_measurements = []
        for measurement in gpu_metrics_list:
            if measurement:
                all_measurements.extend(measurement)
        
        if not all_measurements:
            return []
        
        # Group by node and GPU
        gpu_groups = {}
        for metric in all_measurements:
            node_id = metric['node_id']
            gpu_id = metric['gpu_id']
            key = f"{node_id}-{gpu_id}"
            
            if key not in gpu_groups:
                gpu_groups[key] = []
            gpu_groups[key].append(metric)
        
        # Calculate averages untuk setiap GPU
        averaged_gpus = []
        for key, gpu_data in gpu_groups.items():
            if not gpu_data:
                continue
                
            first_gpu = gpu_data[0]
            
            # Calculate averages
            avg_gpu = {
                'node_id': first_gpu['node_id'],
                'node_ip': first_gpu['node_ip'],
                'gpu_id': first_gpu['gpu_id'],
                'gpu_name': first_gpu['gpu_name'],
                'gpu_memory_total': first_gpu['gpu_memory_total'],
                'avg_gpu_load': np.mean([g['gpu_load'] for g in gpu_data]),
                'avg_gpu_memory_percent': np.mean([g['gpu_memory_percent'] for g in gpu_data]),
                'avg_gpu_memory_used': np.mean([g['gpu_memory_used'] for g in gpu_data]),
                'avg_gpu_temperature': np.mean([g['gpu_temperature'] for g in gpu_data if g['gpu_temperature'] is not None]) if any(g['gpu_temperature'] is not None for g in gpu_data) else None,
                'std_gpu_load': np.std([g['gpu_load'] for g in gpu_data]),
                'std_gpu_memory_percent': np.std([g['gpu_memory_percent'] for g in gpu_data]),
                'measurements_count': len(gpu_data)
            }
            
            averaged_gpus.append(avg_gpu)
        
        return averaged_gpus

    def connect_to_cluster(self):
        """Connect ke Ray cluster dan setup monitoring"""
        logger.info(f"Connecting to Ray cluster: {self.ray_address or 'auto-detect'}")

        try:
            if self.ray_address:
                ray.init(address=self.ray_address, ignore_reinit_error=True)
            else:
                ray.init(address='auto', ignore_reinit_error=True)

            self.connected = True
            logger.info("Connected to Ray cluster")
            
            # Get cluster information
            try:
                self.cluster_info = {
                    "cluster_resources": ray.cluster_resources(),
                    "available_resources": ray.available_resources(),
                    "nodes": len(ray.nodes()),
                    "alive_nodes": len([n for n in ray.nodes() if n.get('Alive', False)])
                }
                logger.info(f"Cluster has {self.cluster_info['alive_nodes']} alive nodes out of {self.cluster_info['nodes']} total")
            except Exception as e:
                logger.warning(f"Could not get cluster info: {e}")
            
            # Setup GPU monitors
            if not self.setup_gpu_monitors():
                logger.warning("Failed to setup GPU monitors, falling back to local monitoring")
                return True  # Still continue even if GPU monitoring fails
            
            # Get cluster GPU information
            if not self.get_cluster_gpu_info():
                logger.warning("Failed to get cluster GPU info")
            
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
        """Test single inference dengan detailed multi-node multi-GPU monitoring"""
        if not self.deployment_available:
            logger.error("No deployment available")
            return None

        logger.info(f"Testing inference ({category or 'unknown'}): {prompt[:40]}...")

        # Collect cluster metrics before
        cluster_gpu_before = self.collect_cluster_gpu_metrics()
        cluster_system_before = self.collect_cluster_system_metrics()

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
            cluster_gpu_after = self.collect_cluster_gpu_metrics()
            cluster_system_after = self.collect_cluster_system_metrics()

            if result and "error" not in result:
                throughput = result['completion_tokens'] / total_time if total_time > 0 else 0

                # Analyze response characteristics
                response_analysis = self.analyze_response_characteristics(result['text'])

                # Calculate cluster averages
                all_gpu_metrics = cluster_gpu_before + cluster_gpu_after
                avg_cluster_gpu_load = np.mean([gpu['gpu_load'] for gpu in all_gpu_metrics]) if all_gpu_metrics else 0
                avg_cluster_gpu_memory = np.mean([gpu['gpu_memory_percent'] for gpu in all_gpu_metrics]) if all_gpu_metrics else 0
                
                all_system_metrics = cluster_system_before + cluster_system_after
                avg_cluster_cpu = np.mean([sys['cpu_percent'] for sys in all_system_metrics]) if all_system_metrics else 0
                avg_cluster_ram = np.mean([sys['memory_percent'] for sys in all_system_metrics]) if all_system_metrics else 0

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
                    "cluster_cpu_usage": avg_cluster_cpu,
                    "cluster_ram_usage": avg_cluster_ram,
                    "cluster_gpu_usage": avg_cluster_gpu_load,
                    "cluster_gpu_memory": avg_cluster_gpu_memory,
                    "timestamp": time.time(),
                    # Detailed cluster metrics
                    "cluster_gpu_metrics_before": cluster_gpu_before,
                    "cluster_gpu_metrics_after": cluster_gpu_after,
                    "cluster_system_metrics_before": cluster_system_before,
                    "cluster_system_metrics_after": cluster_system_after,
                    "total_cluster_gpus": len(cluster_gpu_before) if cluster_gpu_before else 0,
                    "cluster_nodes": len(set([gpu['node_id'] for gpu in cluster_gpu_before])) if cluster_gpu_before else 0
                }

                logger.info(f"✓ Success: {total_time:.2f}s, {throughput:.1f} tokens/s, "
                          f"{response_analysis['word_count']} words ({response_analysis['response_length_category']})")
                
                if cluster_gpu_before:
                    # Group by node for logging
                    node_gpu_summary = {}
                    for gpu in cluster_gpu_after:
                        node_key = f"{gpu['node_ip']}"
                        if node_key not in node_gpu_summary:
                            node_gpu_summary[node_key] = []
                        node_gpu_summary[node_key].append(f"GPU{gpu['gpu_id']}={gpu['gpu_load']:.1f}%")
                    
                    for node_ip, gpu_loads in node_gpu_summary.items():
                        logger.info(f"  Node {node_ip}: {', '.join(gpu_loads)}")
                
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
        all_cluster_gpu_metrics = []

        for i in range(iterations):
            result = await self.test_inference(prompt, max_tokens, temperature, category)
            if result:
                results.append(result)
                
                # Collect cluster GPU metrics for averaging
                if result.get('cluster_gpu_metrics_before'):
                    all_cluster_gpu_metrics.append(result['cluster_gpu_metrics_before'])
                if result.get('cluster_gpu_metrics_after'):
                    all_cluster_gpu_metrics.append(result['cluster_gpu_metrics_after'])
                
                logger.info(f"  Iteration {i+1}: {result['total_time']:.2f}s, "
                            f"{result['throughput']:.1f} tok/s, "
                            f"{result['response_analysis']['word_count']} words, "
                            f"{result['total_cluster_gpus']} GPUs across {result['cluster_nodes']} nodes")

            # Small delay between iterations
            if i < iterations - 1:
                await asyncio.sleep(0.5)

        if results:
            # Calculate cluster GPU averages across all iterations
            averaged_cluster_gpu_metrics = self.calculate_cluster_gpu_averages(all_cluster_gpu_metrics)
            
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
                # Cluster information
                "total_cluster_gpus": results[0].get('total_cluster_gpus', 0),
                "cluster_nodes": results[0].get('cluster_nodes', 0),
                "cluster_gpu_metrics": averaged_cluster_gpu_metrics,
                "cluster_gpu_info": self.cluster_gpu_info
            }

            logger.info(f"Average results: {avg_result['avg_total_time']:.2f}s, "
                        f"{avg_result['avg_throughput']:.1f} tok/s, "
                        f"{avg_word_count:.0f} words")
            
            # Log cluster GPU averages
            if averaged_cluster_gpu_metrics:
                # Group by node
                node_summary = {}
                for gpu in averaged_cluster_gpu_metrics:
                    node_key = gpu['node_ip']
                    if node_key not in node_summary:
                        node_summary[node_key] = []
                    node_summary[node_key].append(f"{gpu['gpu_name']}: {gpu['avg_gpu_load']:.1f}% load, {gpu['avg_gpu_memory_percent']:.1f}% mem")
                
                for node_ip, gpu_summaries in node_summary.items():
                    logger.info(f"Node {node_ip} averages: {'; '.join(gpu_summaries)}")
            
            return avg_result

        return None

    async def run_batch_evaluation(self, prompts: List, iterations: int = 3,
                                  max_tokens: int = 100, temperature: float = 0.7):
        """Run evaluation untuk multiple prompts"""
        total_gpus = len(self.cluster_gpu_info)
        total_nodes = len(set([gpu['node_id'] for gpu in self.cluster_gpu_info])) if self.cluster_gpu_info else 0
        
        logger.info(f"Running batch evaluation: {len(prompts)} prompts x {iterations} iterations")
        logger.info(f"Cluster: {total_gpus} GPUs across {total_nodes} nodes")

        all_results = []

        for i, prompt_data in enumerate(prompts):
            category = prompt_data.get('category', 'uncategorized') if isinstance(prompt_data, dict) else 'uncategorized'
            logger.info(f"\nPrompt {i+1}/{len(prompts)} ({category})")

            result = await self.run_single_prompt_evaluation(
                prompt_data, iterations, max_tokens, temperature
            )

            if result:
                all_results.append(result)

            # Delay between different prompts
            if i < len(prompts) - 1:
                await asyncio.sleep(1)

        return all_results
    def generate_comprehensive_report(self, results: List[Dict], output_file: str = None):
        """Generate comprehensive evaluation report"""
        if not results:
            logger.warning("No results to generate report")
            return None

        # Calculate overall statistics
        total_prompts = len(results)
        total_iterations = sum([r['iterations'] for r in results])
        
        # Group results by category
        category_results = {}
        for result in results:
            category = result.get('prompt_category', 'uncategorized')
            if category not in category_results:
                category_results[category] = []
            category_results[category].append(result)
        
        # Calculate category averages
        category_stats = {}
        for category, cat_results in category_results.items():
            category_stats[category] = {
                'count': len(cat_results),
                'avg_time': np.mean([r['avg_total_time'] for r in cat_results]),
                'avg_throughput': np.mean([r['avg_throughput'] for r in cat_results]),
                'avg_tokens': np.mean([r['avg_completion_tokens'] for r in cat_results]),
                'avg_words': np.mean([r['avg_response_analysis']['avg_word_count'] for r in cat_results]),
                'avg_gpu_usage': np.mean([r['avg_cluster_gpu'] for r in cat_results]),
                'avg_gpu_memory': np.mean([r['avg_cluster_gpu_mem'] for r in cat_results])
            }
        
        # Overall performance metrics
        overall_stats = {
            'total_prompts_tested': total_prompts,
            'total_iterations': total_iterations,
            'avg_response_time': np.mean([r['avg_total_time'] for r in results]),
            'avg_throughput': np.mean([r['avg_throughput'] for r in results]),
            'avg_tokens_per_response': np.mean([r['avg_completion_tokens'] for r in results]),
            'avg_words_per_response': np.mean([r['avg_response_analysis']['avg_word_count'] for r in results]),
            'avg_cluster_gpu_usage': np.mean([r['avg_cluster_gpu'] for r in results]),
            'avg_cluster_gpu_memory': np.mean([r['avg_cluster_gpu_mem'] for r in results]),
            'avg_cluster_cpu_usage': np.mean([r['avg_cluster_cpu'] for r in results]),
            'avg_cluster_ram_usage': np.mean([r['avg_cluster_ram'] for r in results])
        }
        
        # Performance consistency metrics
        time_variability = np.std([r['avg_total_time'] for r in results])
        throughput_variability = np.std([r['avg_throughput'] for r in results])
        
        report = {
            'evaluation_summary': {
                'timestamp': time.time(),
                'cluster_info': self.cluster_info,
                'total_cluster_gpus': results[0].get('total_cluster_gpus', 0) if results else 0,
                'cluster_nodes': results[0].get('cluster_nodes', 0) if results else 0,
                'cluster_gpu_info': self.cluster_gpu_info
            },
            'overall_performance': overall_stats,
            'performance_consistency': {
                'time_std_dev': time_variability,
                'throughput_std_dev': throughput_variability,
                'consistency_score': 1.0 / (1.0 + time_variability)  # Higher is better
            },
            'category_breakdown': category_stats,
            'detailed_results': results,
            'recommendations': self.generate_recommendations(results, overall_stats, category_stats)
        }
        
        # Save report if output file specified
        if output_file:
            try:
                with open(output_file, 'w') as f:
                    json.dump(report, f, indent=2, default=str)
                logger.info(f"Report saved to {output_file}")
            except Exception as e:
                logger.error(f"Failed to save report: {e}")
        
        return report
    
    def generate_recommendations(self, results: List[Dict], overall_stats: Dict, category_stats: Dict):
        """Generate performance recommendations based on results"""
        recommendations = []
        
        # Check overall performance
        avg_throughput = overall_stats['avg_throughput']
        avg_gpu_usage = overall_stats['avg_cluster_gpu_usage']
        avg_gpu_memory = overall_stats['avg_cluster_gpu_memory']
        
        if avg_throughput < 10:
            recommendations.append({
                'type': 'performance',
                'priority': 'high',
                'message': f'Low throughput detected ({avg_throughput:.1f} tokens/s). Consider optimizing model or increasing resources.'
            })
        
        if avg_gpu_usage < 30:
            recommendations.append({
                'type': 'resource_utilization',
                'priority': 'medium',
                'message': f'Low GPU utilization ({avg_gpu_usage:.1f}%). Consider increasing batch size or concurrent requests.'
            })
        
        if avg_gpu_memory > 90:
            recommendations.append({
                'type': 'resource_utilization',
                'priority': 'high',
                'message': f'High GPU memory usage ({avg_gpu_memory:.1f}%). Risk of OOM errors.'
            })
        
        # Check category performance variations
        category_throughputs = [stats['avg_throughput'] for stats in category_stats.values()]
        if len(category_throughputs) > 1:
            throughput_variation = np.std(category_throughputs) / np.mean(category_throughputs)
            if throughput_variation > 0.3:
                recommendations.append({
                    'type': 'optimization',
                    'priority': 'medium',
                    'message': f'High performance variation across prompt categories ({throughput_variation:.1%}). Consider prompt-specific optimization.'
                })
        
        # Check for resource imbalance
        if len(self.cluster_gpu_info) > 1:
            recommendations.append({
                'type': 'scaling',
                'priority': 'info',
                'message': f'Multi-GPU cluster detected ({len(self.cluster_gpu_info)} GPUs). Monitor load balancing across GPUs.'
            })
        
        return recommendations
    
    def print_summary_report(self, report: Dict):
        """Print a human-readable summary report"""
        print("\n" + "="*80)
        print("RAY MODEL EVALUATION SUMMARY REPORT")
        print("="*80)
        
        # Cluster information
        cluster_info = report['evaluation_summary']
        print(f"\nCluster Information:")
        print(f"  Total GPUs: {cluster_info['total_cluster_gpus']}")
        print(f"  Total Nodes: {cluster_info['cluster_nodes']}")
        print(f"  Cluster Resources: {cluster_info.get('cluster_info', {}).get('cluster_resources', 'N/A')}")
        
        # Overall performance
        overall = report['overall_performance']
        print(f"\nOverall Performance:")
        print(f"  Total Prompts: {overall['total_prompts_tested']}")
        print(f"  Total Iterations: {overall['total_iterations']}")
        print(f"  Average Response Time: {overall['avg_response_time']:.3f}s")
        print(f"  Average Throughput: {overall['avg_throughput']:.1f} tokens/s")
        print(f"  Average Tokens/Response: {overall['avg_tokens_per_response']:.0f}")
        print(f"  Average Words/Response: {overall['avg_words_per_response']:.0f}")
        
        # Resource utilization
        print(f"\nResource Utilization:")
        print(f"  Average GPU Usage: {overall['avg_cluster_gpu_usage']:.1f}%")
        print(f"  Average GPU Memory: {overall['avg_cluster_gpu_memory']:.1f}%")
        print(f"  Average CPU Usage: {overall['avg_cluster_cpu_usage']:.1f}%")
        print(f"  Average RAM Usage: {overall['avg_cluster_ram_usage']:.1f}%")
        
        # Performance consistency
        consistency = report['performance_consistency']
        print(f"\nPerformance Consistency:")
        print(f"  Time Std Dev: {consistency['time_std_dev']:.3f}s")
        print(f"  Throughput Std Dev: {consistency['throughput_std_dev']:.1f} tokens/s")
        print(f"  Consistency Score: {consistency['consistency_score']:.3f}")
        
        # Category breakdown
        print(f"\nCategory Performance Breakdown:")
        for category, stats in report['category_breakdown'].items():
            print(f"  {category.upper()}:")
            print(f"    Count: {stats['count']}")
            print(f"    Avg Time: {stats['avg_time']:.3f}s")
            print(f"    Avg Throughput: {stats['avg_throughput']:.1f} tokens/s")
            print(f"    Avg Words: {stats['avg_words']:.0f}")
            print(f"    Avg GPU Usage: {stats['avg_gpu_usage']:.1f}%")
        
        # Recommendations
        print(f"\nRecommendations:")
        for rec in report['recommendations']:
            priority_symbol = "🔴" if rec['priority'] == 'high' else "🟡" if rec['priority'] == 'medium' else "ℹ️"
            print(f"  {priority_symbol} [{rec['type'].upper()}] {rec['message']}")
        
        print("\n" + "="*80)

async def main():
    """Main function untuk menjalankan evaluasi"""
    parser = argparse.ArgumentParser(description='Ray Model Evaluator with Multi-Node Multi-GPU Support')
    parser.add_argument('--ray-address', type=str, help='Ray cluster address (e.g., ray://head-node:10001)')
    parser.add_argument('--iterations', type=int, default=3, help='Number of iterations per prompt')
    parser.add_argument('--max-tokens', type=int, default=100, help='Maximum tokens per response')
    parser.add_argument('--temperature', type=float, default=0.7, help='Temperature for generation')
    parser.add_argument('--category', type=str, choices=['short_text', 'long_text', 'creative', 'technical', 'reasoning', 'mixed', 'all'], 
                        default='mixed', help='Prompt category to test')
    parser.add_argument('--prompts-per-category', type=int, default=2, help='Number of prompts per category for mixed testing')
    parser.add_argument('--output', type=str, help='Output file for detailed report (JSON)')
    parser.add_argument('--custom-prompts', type=str, help='Path to custom prompts JSON file')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = MultiNodeMultiGPUModelEvaluator(ray_address=args.ray_address)
    
    # Connect to cluster
    if not evaluator.connect_to_cluster():
        logger.error("Failed to connect to Ray cluster")
        return
    
    # Check deployment
    if not evaluator.check_deployment():
        logger.error("Model deployment not found")
        return
    
    # Prepare prompts
    prompts = []
    
    if args.custom_prompts:
        # Load custom prompts
        try:
            with open(args.custom_prompts, 'r') as f:
                custom_prompts = json.load(f)
                prompts = custom_prompts if isinstance(custom_prompts, list) else [custom_prompts]
                logger.info(f"Loaded {len(prompts)} custom prompts")
        except Exception as e:
            logger.error(f"Failed to load custom prompts: {e}")
            return
    else:
        # Use built-in prompts
        if args.category == 'mixed':
            prompts = PromptCategories.get_mixed_prompts(args.prompts_per_category)
        elif args.category == 'all':
            prompts = PromptCategories.get_all_prompts()[:args.prompts_per_category * 3]  # Get more prompts
            
        else:
            categories = PromptCategories.get_all_categories()
            if args.category in categories:
                # Get All prompts for the specified category
                category_prompts = categories[args.category]
                prompts = [{'prompt': p, 'category': args.category} for p in category_prompts]
    
    if not prompts:
        logger.error("No prompts to evaluate")
        return
    
    logger.info(f"Starting evaluation with {len(prompts)} prompts")
    
    # Run evaluation
    results = await evaluator.run_batch_evaluation(
        prompts, 
        iterations=args.iterations,
        max_tokens=args.max_tokens,
        temperature=args.temperature
    )
    
    if results:
        # Generate comprehensive report
        report = evaluator.generate_comprehensive_report(results, args.output)
        
        # Print summary
        evaluator.print_summary_report(report)
        
        logger.info(f"Evaluation completed successfully. Tested {len(results)} prompts with {args.iterations} iterations each.")
    else:
        logger.error("No successful evaluations completed")

if __name__ == "__main__":
    asyncio.run(main())