#!/usr/bin/env python3
"""
Ray Model Evaluator - Single GPU Version with Categorized Prompts
Script untuk evaluasi model yang sudah di-deploy di Ray cluster dengan dukungan single GPU
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
from datetime import datetime

import ray
from ray import serve

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

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

class SingleGPUModelEvaluator:
    """Evaluator untuk model yang sudah di-deploy dengan dukungan single GPU"""

    def __init__(self, ray_address=None):
        self.ray_address = ray_address
        self.connected = False
        self.deployment_available = False
        self.gpu_available = False
        self.gpu_info = None

    def get_gpu_info(self):
        """Get single GPU information"""
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                # Use first available GPU
                gpu = gpus[0]
                self.gpu_info = {
                    'id': gpu.id,
                    'name': gpu.name,
                    'total_memory': gpu.memoryTotal,
                    'driver': gpu.driver,
                    'uuid': gpu.uuid
                }
                self.gpu_available = True
                logger.info(f"GPU detected: {gpu.name} ({gpu.memoryTotal}MB)")
                return True
            else:
                logger.warning("No GPU detected")
                self.gpu_available = False
                return False
        except Exception as e:
            logger.warning(f"Could not get GPU info: {e}")
            self.gpu_available = False
            return False

    def collect_gpu_metrics(self):
        """Collect metrics from single GPU"""
        gpu_metrics = {}
        
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Use first GPU
                gpu_metrics = {
                    'gpu_id': gpu.id,
                    'gpu_name': gpu.name,
                    'gpu_load': gpu.load * 100,  # Convert to percentage
                    'gpu_memory_used': gpu.memoryUsed,
                    'gpu_memory_total': gpu.memoryTotal,
                    'gpu_memory_percent': (gpu.memoryUsed / gpu.memoryTotal * 100) if gpu.memoryTotal > 0 else 0,
                    'gpu_temperature': gpu.temperature if hasattr(gpu, 'temperature') else None
                }
            else:
                # Return empty metrics if no GPU
                gpu_metrics = {
                    'gpu_id': 0,
                    'gpu_name': 'No_GPU',
                    'gpu_load': 0,
                    'gpu_memory_used': 0,
                    'gpu_memory_total': 0,
                    'gpu_memory_percent': 0,
                    'gpu_temperature': None
                }
                
        except Exception as e:
            logger.warning(f"Error collecting GPU metrics: {e}")
            gpu_metrics = {
                'gpu_id': 0,
                'gpu_name': 'Error_GPU',
                'gpu_load': 0,
                'gpu_memory_used': 0,
                'gpu_memory_total': 0,
                'gpu_memory_percent': 0,
                'gpu_temperature': None
            }
        
        return gpu_metrics

    def calculate_gpu_averages(self, gpu_metrics_list: List[Dict]):
        """Calculate average metrics for single GPU across time"""
        if not gpu_metrics_list:
            return {}
        
        # Collect all measurements
        gpu_loads = []
        gpu_memory_percents = []
        gpu_memory_used = []
        gpu_temperatures = []
        
        gpu_info = None
        
        for measurement in gpu_metrics_list:
            gpu_loads.append(measurement['gpu_load'])
            gpu_memory_percents.append(measurement['gpu_memory_percent'])
            gpu_memory_used.append(measurement['gpu_memory_used'])
            
            if measurement['gpu_temperature'] is not None:
                gpu_temperatures.append(measurement['gpu_temperature'])
            
            if gpu_info is None:
                gpu_info = {
                    'gpu_id': measurement['gpu_id'],
                    'gpu_name': measurement['gpu_name'],
                    'gpu_memory_total': measurement['gpu_memory_total']
                }
        
        # Calculate averages
        avg_gpu = {
            'gpu_id': gpu_info['gpu_id'] if gpu_info else 0,
            'gpu_name': gpu_info['gpu_name'] if gpu_info else 'Unknown_GPU',
            'gpu_memory_total': gpu_info['gpu_memory_total'] if gpu_info else 0,
            'avg_gpu_load': np.mean(gpu_loads) if gpu_loads else 0,
            'avg_gpu_memory_percent': np.mean(gpu_memory_percents) if gpu_memory_percents else 0,
            'avg_gpu_memory_used': np.mean(gpu_memory_used) if gpu_memory_used else 0,
            'avg_gpu_temperature': np.mean(gpu_temperatures) if gpu_temperatures else None,
            'std_gpu_load': np.std(gpu_loads) if gpu_loads else 0,
            'std_gpu_memory_percent': np.std(gpu_memory_percents) if gpu_memory_percents else 0,
            'measurements_count': len(gpu_loads)
        }
        
        return avg_gpu

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
            
            # Get GPU information
            self.get_gpu_info()
            
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
        """Test single inference with detailed single GPU monitoring and response analysis"""
        if not self.deployment_available:
            logger.error("No deployment available")
            return None

        logger.info(f"Testing inference ({category or 'unknown'}): {prompt[:40]}...")

        # Collect system metrics before
        cpu_before = psutil.cpu_percent()
        ram_before = psutil.virtual_memory().percent
        gpu_metrics_before = self.collect_gpu_metrics()

        # Time the inference
        start_time = time.time()

        try:
            # Get deployment handle
            handle = serve.get_deployment_handle("vllm-model", "vllm-model")

            # Make inference request
            result = await handle.generate.remote(prompt, max_tokens, temperature)

            end_time = time.time()
            total_time = end_time - start_time

            # Collect system metrics after
            cpu_after = psutil.cpu_percent()
            ram_after = psutil.virtual_memory().percent
            gpu_metrics_after = self.collect_gpu_metrics()

            if result and "error" not in result:
                throughput = result['completion_tokens'] / total_time if total_time > 0 else 0

                # Analyze response characteristics
                response_analysis = self.analyze_response_characteristics(result['text'])

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
                    "client_cpu_usage": (cpu_before + cpu_after) / 2,
                    "client_ram_usage": (ram_before + ram_after) / 2,
                    "client_gpu_usage": (gpu_metrics_before['gpu_load'] + gpu_metrics_after['gpu_load']) / 2,
                    "client_gpu_memory": (gpu_metrics_before['gpu_memory_percent'] + gpu_metrics_after['gpu_memory_percent']) / 2,
                    "timestamp": time.time(),
                    # Single GPU metrics
                    "gpu_metrics_before": gpu_metrics_before,
                    "gpu_metrics_after": gpu_metrics_after,
                    "gpu_available": self.gpu_available
                }

                logger.info(f"✓ Success: {total_time:.2f}s, {throughput:.1f} tokens/s, "
                          f"{response_analysis['word_count']} words ({response_analysis['response_length_category']})")
                if self.gpu_available:
                    logger.info(f"  GPU utilization: {gpu_metrics_after['gpu_load']:.1f}%")
                
                return evaluation_result
            else:
                logger.error(f"Inference failed: {result}")
                return None

        except Exception as e:
            logger.error(f"Inference error: {e}")
            return None

    async def run_single_prompt_evaluation(self, prompt_data, iterations: int = 3,
                                           max_tokens: int = 100, temperature: float = 0.7):
        """Run multiple iterations untuk single prompt dengan detailed GPU tracking"""
        
        # Handle both string and dict prompt formats
        if isinstance(prompt_data, str):
            prompt = prompt_data
            category = "uncategorized"
        else:
            prompt = prompt_data.get('prompt', prompt_data)
            category = prompt_data.get('category', 'uncategorized')
        
        logger.info(f"Evaluating {category} prompt with {iterations} iterations: {prompt[:50]}...")

        results = []
        all_gpu_metrics_before = []
        all_gpu_metrics_after = []

        for i in range(iterations):
            result = await self.test_inference(prompt, max_tokens, temperature, category)
            if result:
                results.append(result)
                
                # Collect GPU metrics for averaging
                if result.get('gpu_metrics_before'):
                    all_gpu_metrics_before.append(result['gpu_metrics_before'])
                if result.get('gpu_metrics_after'):
                    all_gpu_metrics_after.append(result['gpu_metrics_after'])
                
                logger.info(f"  Iteration {i+1}: {result['total_time']:.2f}s, "
                            f"{result['throughput']:.1f} tok/s, "
                            f"{result['response_analysis']['word_count']} words")

            # Small delay between iterations
            if i < iterations - 1:
                await asyncio.sleep(0.5)

        if results:
            # Calculate GPU averages across all iterations
            combined_gpu_metrics = all_gpu_metrics_before + all_gpu_metrics_after
            averaged_gpu_metrics = self.calculate_gpu_averages(combined_gpu_metrics)
            
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
                "avg_client_cpu": np.mean([r['client_cpu_usage'] for r in results]),
                "avg_client_ram": np.mean([r['client_ram_usage'] for r in results]),
                "avg_client_gpu": np.mean([r['client_gpu_usage'] for r in results]),
                "avg_client_gpu_mem": np.mean([r['client_gpu_memory'] for r in results]),
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
                # Single GPU metrics
                "gpu_available": results[0].get('gpu_available', False),
                "gpu_metrics": averaged_gpu_metrics,
                "gpu_info": self.gpu_info
            }

            logger.info(f"Average results: {avg_result['avg_total_time']:.2f}s, "
                        f"{avg_result['avg_throughput']:.1f} tok/s, "
                        f"{avg_word_count:.0f} words")
            
            # Log GPU averages
            if averaged_gpu_metrics and self.gpu_available:
                logger.info(f"GPU averages: {averaged_gpu_metrics['gpu_name']}: "
                           f"{averaged_gpu_metrics['avg_gpu_load']:.1f}% load, "
                           f"{averaged_gpu_metrics['avg_gpu_memory_percent']:.1f}% mem")
            
            return avg_result

        return None

    async def run_batch_evaluation(self, prompts: List, iterations: int = 3,
                                   max_tokens: int = 100, temperature: float = 0.7):
        """Run evaluation untuk multiple prompts"""
        logger.info(f"Running batch evaluation: {len(prompts)} prompts x {iterations} iterations")
        logger.info(f"GPU available: {self.gpu_available}")

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
    def run_category_evaluation(self, category: str = None, iterations: int = 3,
                                max_tokens: int = 100, temperature: float = 0.7):
          """Run evaluation for specific category"""
          categories = PromptCategories.get_all_categories()
          
          if category and category in categories:
              prompts = [{'prompt': p, 'category': category} for p in categories[category]]
              logger.info(f"Running evaluation for {category} category ({len(prompts)} prompts)")
          elif category == 'all':
              prompts = PromptCategories.get_mixed_prompts(count_per_category=len(list(categories.values())[0]))
              logger.info(f"Running evaluation for all categories ({len(prompts)} prompts)")
          else:
              prompts = PromptCategories.get_mixed_prompts(count_per_category=2)
              logger.info(f"Running mixed evaluation ({len(prompts)} prompts)")

          return self.run_evaluation_sync(prompts, iterations, max_tokens, temperature)

    def run_evaluation_sync(self, prompts: List = None, iterations: int = 3,
                            max_tokens: int = 100, temperature: float = 0.7):
        """Synchronous wrapper untuk evaluation"""
        if prompts is None:
            # Use mixed prompts from all categories
            prompts = PromptCategories.get_mixed_prompts(count_per_category=2)

        async def _run_async():
            return await self.run_batch_evaluation(prompts, iterations, max_tokens, temperature)

        # Run async function
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(_run_async())
        except RuntimeError:
            return asyncio.run(_run_async())

    def print_summary(self, results: List[Dict]):
        """Print evaluation summary with detailed GPU information and category analysis"""
        if not results:
            print("No results to summarize")
            return

        print("\n" + "="*80)
        print("SINGLE GPU MODEL EVALUATION SUMMARY WITH CATEGORIZED PROMPTS")
        print("="*80)
        print(f"Ray cluster: {self.ray_address or 'auto-detected'}")
        print(f"Model: {results[0].get('model', 'unknown')}")
        print(f"Total prompts: {len(results)}")
        print(f"Iterations per prompt: {results[0].get('iterations', 'unknown')}")
        print(f"GPU available: {self.gpu_available}")

        # Print GPU information
        if self.gpu_info:
            print(f"\nGPU Information:")
            print(f"  GPU: {self.gpu_info['name']} ({self.gpu_info['total_memory']}MB)")

        # Category-based analysis
        print(f"\nPrompt Category Analysis:")
        category_stats = {}
        for result in results:
            category = result.get('prompt_category', 'uncategorized')
            if category not in category_stats:
                category_stats[category] = []
            category_stats[category].append(result)

        for category, cat_results in category_stats.items():
            avg_time = np.mean([r['avg_total_time'] for r in cat_results])
            avg_throughput = np.mean([r['avg_throughput'] for r in cat_results])
            avg_words = np.mean([r['avg_response_analysis']['avg_word_count'] for r in cat_results])
            
            print(f"  {category.upper()}: {len(cat_results)} prompts")
            print(f"    Avg time: {avg_time:.2f}s, Avg throughput: {avg_throughput:.1f} tok/s, Avg words: {avg_words:.0f}")

        # Calculate overall statistics
        avg_time = np.mean([r['avg_total_time'] for r in results])
        avg_throughput = np.mean([r['avg_throughput'] for r in results])
        avg_prompt_tokens = np.mean([r['avg_prompt_tokens'] for r in results])
        avg_completion_tokens = np.mean([r['avg_completion_tokens'] for r in results])
        avg_cpu = np.mean([r['avg_client_cpu'] for r in results])
        avg_ram = np.mean([r['avg_client_ram'] for r in results])
        avg_words = np.mean([r['avg_response_analysis']['avg_word_count'] for r in results])

        # Calculate variability
        std_time = np.std([r['avg_total_time'] for r in results])
        std_throughput = np.std([r['avg_throughput'] for r in results])

        print(f"\nOverall Performance Metrics:")
        print(f"  Average inference time: {avg_time:.2f}s (±{std_time:.2f}s)")
        print(f"  Average throughput: {avg_throughput:.2f} tokens/s (±{std_throughput:.2f})")
        print(f"  Average prompt tokens: {avg_prompt_tokens:.0f}")
        print(f"  Average completion tokens: {avg_completion_tokens:.0f}")
        print(f"  Average response words: {avg_words:.0f}")

        print(f"\nClient Resource Usage:")
        print(f"  Average CPU usage: {avg_cpu:.1f}%")
        print(f"  Average RAM usage: {avg_ram:.1f}%")

        # Response length distribution
        length_categories = {}
        for result in results:
            for detail in result.get('detailed_results', []):
                length_cat = detail['response_analysis']['response_length_category']
                length_categories[length_cat] = length_categories.get(length_cat, 0) + 1
        
        print(f"\nResponse Length Distribution:")
        for length_cat, count in sorted(length_categories.items()):
            print(f"  {length_cat}: {count} responses")

        # Detailed GPU metrics
        if self.gpu_available and results[0].get('gpu_metrics'):
            print(f"\nDetailed GPU Metrics (averaged across all evaluations):")
            
            # Collect all GPU metrics from all results
            all_gpu_metrics = []
            for result in results:
                if result.get('gpu_metrics'):
                    all_gpu_metrics.append(result['gpu_metrics'])
            
            # Calculate overall GPU averages
            if all_gpu_metrics:
                avg_load = np.mean([g['avg_gpu_load'] for g in all_gpu_metrics])
                avg_mem = np.mean([g['avg_gpu_memory_percent'] for g in all_gpu_metrics])
                avg_mem_used = np.mean([g['avg_gpu_memory_used'] for g in all_gpu_metrics])
                gpu_name = all_gpu_metrics[0]['gpu_name']
                total_mem = all_gpu_metrics[0]['gpu_memory_total']
                
                print(f"  GPU ({gpu_name}):")
                print(f"    Average load: {avg_load:.1f}%")
                print(f"    Average memory usage: {avg_mem:.1f}% ({avg_mem_used:.0f}MB / {total_mem}MB)")
                
                temps = [g.get('avg_gpu_temperature') for g in all_gpu_metrics if g.get('avg_gpu_temperature') is not None]
                if temps:
                    avg_temp = np.mean(temps)
                    print(f"    Average temperature: {avg_temp:.1f}°C")

        print("="*80)

        # Per-prompt breakdown
        print("\nPer-prompt Results (by Category):")
        for category, cat_results in category_stats.items():
            print(f"\n{category.upper()}:")
            for i, result in enumerate(cat_results):
                avg_words = result['avg_response_analysis']['avg_word_count']
                length_cat = result['avg_response_analysis']['dominant_length_category']
                print(f"  {result['prompt'][:50]:50s} | "
                      f"Time: {result['avg_total_time']:5.2f}s | "
                      f"Throughput: {result['avg_throughput']:5.1f} tok/s | "
                      f"Words: {avg_words:3.0f} ({length_cat})")

    def save_results(self, results: List[Dict], filename: str = None):
        """Save evaluation results with detailed GPU information and category analysis"""
        if filename is None:
            timestamp = int(time.time())
            filename = f"categorized_single_gpu_evaluation_{timestamp}.json"

        # Load deployment info if available
        deployment_info = {}
        try:
            with open("deployment_info.json", "r") as f:
                deployment_info = json.load(f)
        except:
            pass

        # Get cluster info
        cluster_info = {}
        try:
            if self.connected:
                cluster_info = {
                    "cluster_resources": ray.cluster_resources(),
                    "available_resources": ray.available_resources(),
                    "nodes": len(ray.nodes())
                }
        except:
            pass

        # Analyze results by category
        category_analysis = {}
        for result in results:
            category = result.get('prompt_category', 'uncategorized')
            if category not in category_analysis:
                category_analysis[category] = {
                    'count': 0,
                    'avg_time': [],
                    'avg_throughput': [],
                    'avg_word_count': [],
                    'avg_tokens': []
                }
            
            cat_data = category_analysis[category]
            cat_data['count'] += 1
            cat_data['avg_time'].append(result['avg_total_time'])
            cat_data['avg_throughput'].append(result['avg_throughput'])
            cat_data['avg_word_count'].append(result['avg_response_analysis']['avg_word_count'])
            cat_data['avg_tokens'].append(result['avg_completion_tokens'])

        # Calculate
        # Lanjutan dari method save_results - bagian calculate category statistics
        
        # Calculate category statistics
        for category, data in category_analysis.items():
            category_analysis[category] = {
                'count': data['count'],
                'avg_time': np.mean(data['avg_time']),
                'avg_throughput': np.mean(data['avg_throughput']), 
                'avg_word_count': np.mean(data['avg_word_count']),
                'avg_tokens': np.mean(data['avg_tokens']),
                'std_time': np.std(data['avg_time']),
                'std_throughput': np.std(data['avg_throughput'])
            }

        # Collect overall GPU metrics
        overall_gpu_metrics = {}
        if self.gpu_available:
            all_gpu_data = []
            for result in results:
                if result.get('gpu_metrics'):
                    all_gpu_data.append(result['gpu_metrics'])
            
            if all_gpu_data:
                overall_gpu_metrics = {
                    'gpu_name': all_gpu_data[0]['gpu_name'],
                    'gpu_total_memory': all_gpu_data[0]['gpu_memory_total'],
                    'overall_avg_load': np.mean([g['avg_gpu_load'] for g in all_gpu_data]),
                    'overall_avg_memory_percent': np.mean([g['avg_gpu_memory_percent'] for g in all_gpu_data]),
                    'overall_avg_memory_used': np.mean([g['avg_gpu_memory_used'] for g in all_gpu_data]),
                    'gpu_info': self.gpu_info
                }

        # Prepare complete results with metadata
        complete_results = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "ray_address": self.ray_address,
                "gpu_available": self.gpu_available,
                "gpu_info": self.gpu_info,
                "total_prompts": len(results),
                "iterations_per_prompt": results[0].get('iterations', 'unknown') if results else 0,
                "evaluation_type": "categorized_single_gpu"
            },
            "deployment_info": deployment_info,
            "cluster_info": cluster_info,
            "category_analysis": category_analysis,
            "overall_gpu_metrics": overall_gpu_metrics,
            "overall_statistics": {
                "avg_inference_time": np.mean([r['avg_total_time'] for r in results]),
                "avg_throughput": np.mean([r['avg_throughput'] for r in results]),
                "avg_prompt_tokens": np.mean([r['avg_prompt_tokens'] for r in results]),
                "avg_completion_tokens": np.mean([r['avg_completion_tokens'] for r in results]),
                "avg_response_words": np.mean([r['avg_response_analysis']['avg_word_count'] for r in results]),
                "std_inference_time": np.std([r['avg_total_time'] for r in results]),
                "std_throughput": np.std([r['avg_throughput'] for r in results])
            },
            "detailed_results": results
        }

        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(complete_results, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to: {filename}")
            return filename
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            return None

    def disconnect(self):
        """Disconnect dari Ray cluster"""
        if self.connected:
            try:
                ray.shutdown()
                self.connected = False
                self.deployment_available = False
                logger.info("Disconnected from Ray cluster")
            except Exception as e:
                logger.warning(f"Error during disconnect: {e}")

def main():
      """Main function"""
      parser = argparse.ArgumentParser(description="Evaluate vLLM model deployed on Ray cluster with multi-GPU support and categorized prompts")
      parser.add_argument("--ray-address", type=str, default=None,
                          help="Ray cluster address")
      parser.add_argument("--prompts-file", type=str, default=None,
                          help="File containing custom prompts")
      parser.add_argument("--category", type=str, default=None,
                          choices=['short_text', 'long_text', 'creative', 'technical', 'reasoning', 'all'],
                          help="Run evaluation for specific category")
      parser.add_argument("--iterations", type=int, default=3,
                          help="Number of iterations per prompt")
      parser.add_argument("--max-tokens", type=int, default=100,
                          help="Maximum tokens to generate")
      parser.add_argument("--temperature", type=float, default=0.7,
                          help="Sampling temperature")
      parser.add_argument("--output", type=str, default=None,
                          help="Output filename for results")
      parser.add_argument("--list-categories", action="store_true",
                          help="List available prompt categories and exit")

      args = parser.parse_args()

      # List categories if requested
      if args.list_categories:
          print("\nAvailable Prompt Categories:")
          print("="*40)
          categories = PromptCategories.get_all_categories()
          for category, prompts in categories.items():
              print(f"\n{category.upper()} ({len(prompts)} prompts):")
              for i, prompt in enumerate(prompts[:3], 1):  # Show first 3 examples
                  print(f"  {i}. {prompt[:60]}{'...' if len(prompt) > 60 else ''}")
              if len(prompts) > 3:
                  print(f"  ... and {len(prompts) - 3} more")
          print(f"\nUse --category <category_name> to run evaluation for specific category")
          print(f"Use --category all to run evaluation for all categories")
          return

      print("Multi-GPU Ray Model Evaluator with Categorized Prompts")
      print("="*55)

      evaluator = SingleGPUModelEvaluator(args.ray_address)

      try:
          # Connect to cluster
          if not evaluator.connect_to_cluster():
              print("Failed to connect to Ray cluster")
              return

          # Check deployment
          if not evaluator.check_deployment():
              print("No model deployment found. Please run the deployer first.")
              return

          # Load prompts
          prompts = None
          if args.prompts_file:
              prompts = evaluator.load_custom_prompts(args.prompts_file)
              if prompts:
                  print(f"Loaded {len(prompts)} prompts from {args.prompts_file}")
              else:
                  print("Failed to load custom prompts, using default prompts")

          # Quick test first
          print("\n=== Quick Test ===")
          quick_result = asyncio.run(evaluator.test_inference("Hello! How are you today?"))
          if quick_result:
              print(f"✓ Quick test successful: {quick_result['total_time']:.2f}s, "
                    f"{quick_result['throughput']:.1f} tokens/s, "
                    f"{quick_result['response_analysis']['word_count']} words")
              if quick_result.get('gpu_count', 0) > 0:
                  print(f"  Detected {quick_result['gpu_count']} GPU(s)")
          else:
              print("✗ Quick test failed")
              return

          # Full evaluation
          print("\n=== Full Evaluation ===")
          
          if prompts:
              # Use custom prompts
              results = evaluator.run_evaluation_sync(
                  prompts=prompts,
                  iterations=args.iterations,
                  max_tokens=args.max_tokens,
                  temperature=args.temperature
              )
          elif args.category:
              # Use category-specific evaluation
              results = evaluator.run_category_evaluation(
                  category=args.category,
                  iterations=args.iterations,
                  max_tokens=args.max_tokens,
                  temperature=args.temperature
              )
          else:
              # Use mixed evaluation
              results = evaluator.run_evaluation_sync(
                  iterations=args.iterations,
                  max_tokens=args.max_tokens,
                  temperature=args.temperature
              )

          if results:
              # Print summary
              evaluator.print_summary(results)

              # Save results
              output_file = evaluator.save_results(results, args.output)
              print(f"\n✓ Evaluation completed! Results saved to {output_file}")
              
              # Print quick category summary
              category_stats = {}
              for result in results:
                  category = result.get('prompt_category', 'uncategorized')
                  if category not in category_stats:
                      category_stats[category] = []
                  category_stats[category].append(result)
              
              print(f"\nQuick Summary by Category:")
              for category, cat_results in category_stats.items():
                  avg_time = np.mean([r['avg_total_time'] for r in cat_results])
                  avg_words = np.mean([r['avg_response_analysis']['avg_word_count'] for r in cat_results])
                  print(f"  {category}: {len(cat_results)} prompts, {avg_time:.2f}s avg, {avg_words:.0f} words avg")
                  
          else:
              print("No results obtained")

      except Exception as e:
          logger.error(f"Evaluation failed: {e}")
          import traceback
          traceback.print_exc()
      finally:
          # Disconnect from cluster
          try:
              ray.disconnect()
          except:
              pass

if __name__ == "__main__":
    main()