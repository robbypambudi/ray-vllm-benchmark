#!/usr/bin/env python3
"""
Ray Model Stress Test Evaluator - Multi GPU Version
Script untuk stress testing model yang sudah di-deploy di Ray cluster dengan dukungan multi GPU
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
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import threading
import random

import ray
from ray import serve

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class StressTestEvaluator:
    """Stress Test Evaluator untuk model yang sudah di-deploy dengan dukungan multi GPU"""

    def __init__(self, ray_address=None):
        self.ray_address = ray_address
        self.connected = False
        self.deployment_available = False
        self.gpu_count = 0
        self.gpu_info = []
        self.stress_test_active = False
        self.stress_test_results = []
        self.monitoring_thread = None
        self.stop_monitoring = threading.Event()

    def get_gpu_info(self):
        """Get detailed GPU information"""
        try:
            gpus = GPUtil.getGPUs()
            self.gpu_count = len(gpus)
            self.gpu_info = []
            
            for i, gpu in enumerate(gpus):
                gpu_data = {
                    'id': gpu.id,
                    'name': gpu.name,
                    'total_memory': gpu.memoryTotal,
                    'driver': gpu.driver,
                    'uuid': gpu.uuid
                }
                self.gpu_info.append(gpu_data)
                logger.info(f"GPU {i}: {gpu.name} ({gpu.memoryTotal}MB)")
            
            return True
        except Exception as e:
            logger.warning(f"Could not get GPU info: {e}")
            self.gpu_count = 0
            return False

    def collect_system_metrics(self):
        """Collect comprehensive system metrics"""
        metrics = {
            'timestamp': time.time(),
            'cpu_percent': psutil.cpu_percent(),
            'cpu_count': psutil.cpu_count(),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_used_gb': psutil.virtual_memory().used / (1024**3),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'disk_io_read_mb': psutil.disk_io_counters().read_bytes / (1024**2) if psutil.disk_io_counters() else 0,
            'disk_io_write_mb': psutil.disk_io_counters().write_bytes / (1024**2) if psutil.disk_io_counters() else 0,
            'network_sent_mb': psutil.net_io_counters().bytes_sent / (1024**2) if psutil.net_io_counters() else 0,
            'network_recv_mb': psutil.net_io_counters().bytes_recv / (1024**2) if psutil.net_io_counters() else 0,
            'gpus': []
        }
        
        # Collect GPU metrics
        try:
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                gpu_data = {
                    'id': gpu.id,
                    'name': gpu.name,
                    'load_percent': gpu.load * 100,
                    'memory_used_mb': gpu.memoryUsed,
                    'memory_total_mb': gpu.memoryTotal,
                    'memory_percent': (gpu.memoryUsed / gpu.memoryTotal * 100) if gpu.memoryTotal > 0 else 0,
                    'temperature': gpu.temperature if hasattr(gpu, 'temperature') else None
                }
                metrics['gpus'].append(gpu_data)
        except Exception as e:
            logger.warning(f"Error collecting GPU metrics: {e}")
        
        return metrics

    def start_continuous_monitoring(self, interval=1.0):
        """Start continuous system monitoring in background thread"""
        def monitor():
            while not self.stop_monitoring.is_set():
                if self.stress_test_active:
                    metrics = self.collect_system_metrics()
                    self.stress_test_results.append(metrics)
                time.sleep(interval)
        
        self.monitoring_thread = threading.Thread(target=monitor, daemon=True)
        self.monitoring_thread.start()
        logger.info("Started continuous monitoring")

    def stop_continuous_monitoring(self):
        """Stop continuous monitoring"""
        self.stop_monitoring.set()
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2)
        logger.info("Stopped continuous monitoring")

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
            status_info = serve.status()
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

    async def single_inference(self, prompt: str, max_tokens: int = 100, temperature: float = 0.7):
        """Single inference call with error handling"""
        try:
            handle = serve.get_deployment_handle("vllm-model", "vllm-model")
            start_time = time.time()
            result = await handle.generate.remote(prompt, max_tokens, temperature)
            end_time = time.time()
            
            if result and "error" not in result:
                return {
                    'success': True,
                    'total_time': end_time - start_time,
                    'prompt_tokens': result.get('prompt_tokens', 0),
                    'completion_tokens': result.get('completion_tokens', 0),
                    'total_tokens': result.get('total_tokens', 0),
                    'response_length': len(result.get('text', '')),
                    'timestamp': start_time
                }
            else:
                return {'success': False, 'error': str(result), 'timestamp': time.time()}
        except Exception as e:
            return {'success': False, 'error': str(e), 'timestamp': time.time()}

    async def concurrent_stress_test(self, concurrent_requests: int = 10, duration_minutes: int = 5,
                                   max_tokens: int = 100, temperature: float = 0.7):
        """Stress test dengan concurrent requests"""
        logger.info(f"Starting concurrent stress test: {concurrent_requests} concurrent requests for {duration_minutes} minutes")
        
        test_prompts = [
            "Explain the concept of artificial intelligence in detail.",
            "Write a comprehensive guide on machine learning algorithms.",
            "Describe the evolution of computer technology over the past decade.",
            "What are the implications of quantum computing for the future?",
            "How does deep learning differ from traditional machine learning?",
            "Explain the role of data science in modern business.",
            "What are the ethical considerations in AI development?",
            "Describe the current state of natural language processing.",
            "How can machine learning be applied to healthcare?",
            "What is the future of autonomous vehicles?",
            "Explain blockchain technology and its applications.",
            "How does computer vision work in practice?",
            "What are the challenges in developing AI systems?",
            "Describe the importance of data privacy in AI.",
            "How can AI be used to solve climate change?"
        ]
        
        end_time = time.time() + (duration_minutes * 60)
        self.stress_test_active = True
        self.stress_test_results = []
        
        # Start monitoring
        self.start_continuous_monitoring(interval=0.5)
        
        results = []
        request_count = 0
        error_count = 0
        
        async def worker():
            nonlocal request_count, error_count
            while time.time() < end_time:
                prompt = random.choice(test_prompts)
                result = await self.single_inference(prompt, max_tokens, temperature)
                results.append(result)
                request_count += 1
                
                if not result['success']:
                    error_count += 1
                
                if request_count % 10 == 0:
                    logger.info(f"Processed {request_count} requests, {error_count} errors")
        
        # Start concurrent workers
        tasks = [worker() for _ in range(concurrent_requests)]
        await asyncio.gather(*tasks)
        
        self.stress_test_active = False
        self.stop_continuous_monitoring()
        
        return {
            'test_type': 'concurrent_stress_test',
            'concurrent_requests': concurrent_requests,
            'duration_minutes': duration_minutes,
            'total_requests': request_count,
            'successful_requests': request_count - error_count,
            'error_count': error_count,
            'error_rate': (error_count / request_count * 100) if request_count > 0 else 0,
            'results': results,
            'system_metrics': self.stress_test_results
        }

    async def throughput_test(self, requests_per_second: int = 5, duration_minutes: int = 3,
                             max_tokens: int = 50, temperature: float = 0.7):
        """Test throughput dengan rate limiting"""
        logger.info(f"Starting throughput test: {requests_per_second} RPS for {duration_minutes} minutes")
        
        test_prompts = [
            "What is machine learning?",
            "Explain neural networks briefly.",
            "How does AI work?",
            "What is deep learning?",
            "Describe computer vision.",
            "What are transformers in AI?",
            "How does NLP work?",
            "What is reinforcement learning?",
            "Explain gradient descent.",
            "What is overfitting?"
        ]
        
        end_time = time.time() + (duration_minutes * 60)
        self.stress_test_active = True
        self.stress_test_results = []
        
        # Start monitoring
        self.start_continuous_monitoring(interval=0.5)
        
        results = []
        request_count = 0
        error_count = 0
        
        interval = 1.0 / requests_per_second
        
        while time.time() < end_time:
            start_batch = time.time()
            
            # Send batch of requests
            batch_tasks = []
            for _ in range(requests_per_second):
                if time.time() >= end_time:
                    break
                prompt = random.choice(test_prompts)
                task = self.single_inference(prompt, max_tokens, temperature)
                batch_tasks.append(task)
            
            # Wait for batch completion
            batch_results = await asyncio.gather(*batch_tasks)
            results.extend(batch_results)
            
            request_count += len(batch_results)
            error_count += sum(1 for r in batch_results if not r['success'])
            
            # Rate limiting
            elapsed = time.time() - start_batch
            if elapsed < 1.0:
                await asyncio.sleep(1.0 - elapsed)
            
            if request_count % 20 == 0:
                logger.info(f"Processed {request_count} requests, {error_count} errors")
        
        self.stress_test_active = False
        self.stop_continuous_monitoring()
        
        return {
            'test_type': 'throughput_test',
            'target_rps': requests_per_second,
            'duration_minutes': duration_minutes,
            'total_requests': request_count,
            'successful_requests': request_count - error_count,
            'error_count': error_count,
            'error_rate': (error_count / request_count * 100) if request_count > 0 else 0,
            'actual_rps': request_count / (duration_minutes * 60),
            'results': results,
            'system_metrics': self.stress_test_results
        }

    async def latency_test(self, request_count: int = 100, max_tokens: int = 200, temperature: float = 0.7):
        """Test latency dengan berbagai ukuran prompt"""
        logger.info(f"Starting latency test: {request_count} sequential requests")
        
        # Variasi panjang prompt
        short_prompts = [
            "Hello!",
            "What is AI?",
            "Explain ML.",
            "How are you?",
            "Define neural network."
        ]
        
        medium_prompts = [
            "Explain artificial intelligence and its applications in modern technology.",
            "What are the main differences between supervised and unsupervised learning?",
            "Describe how neural networks work and their importance in deep learning.",
            "How can machine learning be applied to solve real-world problems?",
            "What are the ethical considerations when developing AI systems?"
        ]
        
        long_prompts = [
            "Provide a comprehensive overview of artificial intelligence, including its history, current applications, and future potential. Discuss the various types of machine learning algorithms, their strengths and weaknesses, and how they are used in different industries. Also, address the ethical implications of AI development and deployment.",
            "Explain the concept of deep learning in detail, including the architecture of neural networks, how backpropagation works, and the role of activation functions. Discuss different types of neural networks such as CNNs, RNNs, and Transformers, and provide examples of their applications in computer vision, natural language processing, and other fields.",
            "Analyze the current state of natural language processing, including recent advances in large language models. Discuss the challenges in developing AI systems that can understand and generate human language, the importance of training data, and the potential risks associated with powerful language models."
        ]
        
        self.stress_test_active = True
        self.stress_test_results = []
        
        # Start monitoring
        self.start_continuous_monitoring(interval=1.0)
        
        results = []
        
        for i in range(request_count):
            # Select prompt type based on request number
            if i % 3 == 0:
                prompt = random.choice(short_prompts)
                prompt_type = 'short'
            elif i % 3 == 1:
                prompt = random.choice(medium_prompts)
                prompt_type = 'medium'
            else:
                prompt = random.choice(long_prompts)
                prompt_type = 'long'
            
            result = await self.single_inference(prompt, max_tokens, temperature)
            result['prompt_type'] = prompt_type
            result['prompt_length'] = len(prompt)
            results.append(result)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Completed {i + 1}/{request_count} requests")
            
            # Small delay between requests
            await asyncio.sleep(0.1)
        
        self.stress_test_active = False
        self.stop_continuous_monitoring()
        
        return {
            'test_type': 'latency_test',
            'total_requests': request_count,
            'successful_requests': sum(1 for r in results if r['success']),
            'error_count': sum(1 for r in results if not r['success']),
            'results': results,
            'system_metrics': self.stress_test_results
        }

    async def memory_stress_test(self, large_token_count: int = 1000, iterations: int = 20):
        """Test memory usage dengan request besar"""
        logger.info(f"Starting memory stress test: {iterations} iterations with {large_token_count} tokens each")
        
        # Long prompt untuk memory stress
        long_prompt = """
        Please provide a very detailed and comprehensive analysis of the following topics:
        1. The history and evolution of artificial intelligence from its inception to current state
        2. Deep learning architectures including CNNs, RNNs, LSTMs, GRUs, and Transformers
        3. Natural language processing techniques and their applications
        4. Computer vision algorithms and their real-world implementations
        5. Reinforcement learning principles and applications
        6. The ethical implications of AI development and deployment
        7. Future trends and potential breakthroughs in AI research
        8. The impact of AI on various industries including healthcare, finance, and transportation
        9. Challenges in AI development such as bias, interpretability, and robustness
        10. The role of data in AI systems and data privacy considerations
        Please provide detailed explanations with examples for each topic.
        """
        
        self.stress_test_active = True
        self.stress_test_results = []
        
        # Start monitoring with higher frequency for memory tracking
        self.start_continuous_monitoring(interval=0.5)
        
        results = []
        
        for i in range(iterations):
            logger.info(f"Memory stress iteration {i+1}/{iterations}")
            
            result = await self.single_inference(long_prompt, large_token_count, 0.7)
            results.append(result)
            
            if not result['success']:
                logger.warning(f"Memory stress test failed at iteration {i+1}: {result.get('error', 'Unknown error')}")
            
            # Small delay to observe memory patterns
            await asyncio.sleep(2)
        
        self.stress_test_active = False
        self.stop_continuous_monitoring()
        
        return {
            'test_type': 'memory_stress_test',
            'large_token_count': large_token_count,
            'iterations': iterations,
            'successful_requests': sum(1 for r in results if r['success']),
            'error_count': sum(1 for r in results if not r['success']),
            'results': results,
            'system_metrics': self.stress_test_results
        }

    def analyze_stress_test_results(self, test_result: Dict):
        """Analyze and summarize stress test results"""
        test_type = test_result['test_type']
        
        print(f"\n{'='*80}")
        print(f"STRESS TEST ANALYSIS: {test_type.upper()}")
        print(f"{'='*80}")
        
        # Basic statistics
        total_requests = test_result.get('total_requests', 0)
        successful_requests = test_result.get('successful_requests', 0)
        error_count = test_result.get('error_count', 0)
        error_rate = test_result.get('error_rate', 0)
        
        print(f"Total Requests: {total_requests}")
        print(f"Successful Requests: {successful_requests}")
        print(f"Error Count: {error_count}")
        print(f"Error Rate: {error_rate:.2f}%")
        
        # Analyze successful requests
        successful_results = [r for r in test_result.get('results', []) if r.get('success', False)]
        
        if successful_results:
            response_times = [r['total_time'] for r in successful_results]
            completion_tokens = [r.get('completion_tokens', 0) for r in successful_results]
            throughputs = [r.get('completion_tokens', 0) / r['total_time'] if r['total_time'] > 0 else 0 
                          for r in successful_results]
            
            print(f"\nResponse Time Statistics:")
            print(f"  Average: {np.mean(response_times):.3f}s")
            print(f"  Median: {np.median(response_times):.3f}s")
            print(f"  Min: {np.min(response_times):.3f}s")
            print(f"  Max: {np.max(response_times):.3f}s")
            print(f"  95th percentile: {np.percentile(response_times, 95):.3f}s")
            print(f"  Standard deviation: {np.std(response_times):.3f}s")
            
            print(f"\nThroughput Statistics:")
            print(f"  Average: {np.mean(throughputs):.2f} tokens/s")
            print(f"  Median: {np.median(throughputs):.2f} tokens/s")
            print(f"  Min: {np.min(throughputs):.2f} tokens/s")
            print(f"  Max: {np.max(throughputs):.2f} tokens/s")
            
            if test_type == 'throughput_test':
                actual_rps = test_result.get('actual_rps', 0)
                target_rps = test_result.get('target_rps', 0)
                print(f"\nThroughput Test Specific:")
                print(f"  Target RPS: {target_rps}")
                print(f"  Actual RPS: {actual_rps:.2f}")
                print(f"  RPS Achievement: {(actual_rps/target_rps*100):.1f}%" if target_rps > 0 else "N/A")
        
        # Analyze system metrics
        system_metrics = test_result.get('system_metrics', [])
        if system_metrics:
            print(f"\nSystem Resource Usage During Test:")
            
            cpu_usage = [m['cpu_percent'] for m in system_metrics]
            memory_usage = [m['memory_percent'] for m in system_metrics]
            
            print(f"  CPU Usage:")
            print(f"    Average: {np.mean(cpu_usage):.1f}%")
            print(f"    Max: {np.max(cpu_usage):.1f}%")
            print(f"    Min: {np.min(cpu_usage):.1f}%")
            
            print(f"  Memory Usage:")
            print(f"    Average: {np.mean(memory_usage):.1f}%")
            print(f"    Max: {np.max(memory_usage):.1f}%")
            print(f"    Min: {np.min(memory_usage):.1f}%")
            
            # GPU metrics
            if system_metrics[0].get('gpus'):
                gpu_count = len(system_metrics[0]['gpus'])
                print(f"\n  GPU Metrics ({gpu_count} GPUs):")
                
                for gpu_id in range(gpu_count):
                    gpu_loads = [m['gpus'][gpu_id]['load_percent'] for m in system_metrics 
                                if gpu_id < len(m['gpus'])]
                    gpu_memory = [m['gpus'][gpu_id]['memory_percent'] for m in system_metrics 
                                 if gpu_id < len(m['gpus'])]
                    gpu_temps = [m['gpus'][gpu_id]['temperature'] for m in system_metrics 
                                if gpu_id < len(m['gpus']) and m['gpus'][gpu_id]['temperature'] is not None]
                    
                    gpu_name = system_metrics[0]['gpus'][gpu_id]['name']
                    
                    print(f"    GPU {gpu_id} ({gpu_name}):")
                    print(f"      Load - Avg: {np.mean(gpu_loads):.1f}%, Max: {np.max(gpu_loads):.1f}%")
                    print(f"      Memory - Avg: {np.mean(gpu_memory):.1f}%, Max: {np.max(gpu_memory):.1f}%")
                    if gpu_temps:
                        print(f"      Temperature - Avg: {np.mean(gpu_temps):.1f}°C, Max: {np.max(gpu_temps):.1f}°C")
        
        print(f"{'='*80}")
        
        return {
            'summary': {
                'test_type': test_type,
                'total_requests': total_requests,
                'success_rate': (successful_requests / total_requests * 100) if total_requests > 0 else 0,
                'avg_response_time': np.mean([r['total_time'] for r in successful_results]) if successful_results else 0,
                'avg_throughput': np.mean([r.get('completion_tokens', 0) / r['total_time'] if r['total_time'] > 0 else 0 
                                         for r in successful_results]) if successful_results else 0,
                'p95_response_time': np.percentile([r['total_time'] for r in successful_results], 95) if successful_results else 0,
                'max_cpu_usage': np.max([m['cpu_percent'] for m in system_metrics]) if system_metrics else 0,
                'max_memory_usage': np.max([m['memory_percent'] for m in system_metrics]) if system_metrics else 0,
                'max_gpu_usage': np.max([np.mean([gpu['load_percent'] for gpu in m['gpus']]) 
                                       for m in system_metrics if m.get('gpus')]) if system_metrics else 0
            }
        }

    def save_stress_test_results(self, test_result: Dict, filename: str = None):
        """Save stress test results to file"""
        if filename is None:
            timestamp = int(time.time())
            test_type = test_result.get('test_type', 'stress_test')
            filename = f"{test_type}_results_{timestamp}.json"
        
        # Add system info
        enhanced_result = {
            'test_timestamp': time.time(),
            'test_datetime': datetime.now().isoformat(),
            'gpu_count': self.gpu_count,
            'gpu_info': self.gpu_info,
            'system_info': {
                'cpu_count': psutil.cpu_count(),
                'total_memory_gb': psutil.virtual_memory().total / (1024**3),
            },
            **test_result
        }
        
        with open(filename, 'w') as f:
            json.dump(enhanced_result, f, indent=2)
        
        logger.info(f"Stress test results saved to {filename}")
        return filename

    def generate_performance_report(self, test_results: List[Dict]):
        """Generate comprehensive performance report"""
        print(f"\n{'='*100}")
        print("COMPREHENSIVE PERFORMANCE REPORT")
        print(f"{'='*100}")
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"GPU Count: {self.gpu_count}")
        if self.gpu_info:
            print("GPU Information:")
            for i, gpu in enumerate(self.gpu_info):
                print(f"  GPU {i}: {gpu['name']} ({gpu['total_memory']}MB)")
        
        print(f"\n{'='*100}")
        print("TEST SUMMARY")
        print(f"{'='*100}")
        
        summary_table = []
        for test_result in test_results:
            analysis = self.analyze_stress_test_results(test_result)
            summary = analysis['summary']
            summary_table.append(summary)
        
        # Print summary table
        if summary_table:
            headers = ['Test Type', 'Requests', 'Success Rate', 'Avg Response', 'P95 Response', 'Avg Throughput', 'Max CPU', 'Max Memory', 'Max GPU']
            print(f"{'Test Type':<20} {'Requests':<10} {'Success %':<10} {'Avg Resp(s)':<12} {'P95 Resp(s)':<12} {'Throughput':<12} {'Max CPU%':<10} {'Max Mem%':<10} {'Max GPU%':<10}")
            print("-" * 120)
            
            for summary in summary_table:
                print(f"{summary['test_type']:<20} {summary['total_requests']:<10} {summary['success_rate']:<10.1f} {summary['avg_response_time']:<12.3f} {summary['p95_response_time']:<12.3f} {summary['avg_throughput']:<12.2f} {summary['max_cpu_usage']:<10.1f} {summary['max_memory_usage']:<10.1f} {summary['max_gpu_usage']:<10.1f}")
        
        print(f"\n{'='*100}")
        print("RECOMMENDATIONS")
        print(f"{'='*100}")
        
        # Generate recommendations based on results
        recommendations = []
        
        for summary in summary_table:
            test_type = summary['test_type']
            success_rate = summary['success_rate']
            avg_response = summary['avg_response_time']
            max_gpu = summary['max_gpu_usage']
            max_cpu = summary['max_cpu_usage']
            max_memory = summary['max_memory_usage']
            
            if success_rate < 95:
                recommendations.append(f"⚠️  {test_type}: Low success rate ({success_rate:.1f}%) - Check error logs and resource allocation")
            
            if avg_response > 5.0:
                recommendations.append(f"⚠️  {test_type}: High average response time ({avg_response:.2f}s) - Consider optimizing model or scaling resources")
            
            if max_gpu < 70:
                recommendations.append(f"💡 {test_type}: GPU utilization is low ({max_gpu:.1f}%) - Could handle higher load or use fewer GPUs")
            elif max_gpu > 95:
                recommendations.append(f"⚠️  {test_type}: GPU utilization very high ({max_gpu:.1f}%) - Risk of bottleneck, consider adding more GPUs")
            
            if max_cpu > 90:
                recommendations.append(f"⚠️  {test_type}: High CPU usage ({max_cpu:.1f}%) - CPU bottleneck detected")
            
            if max_memory > 90:
                recommendations.append(f"⚠️  {test_type}: High memory usage ({max_memory:.1f}%) - Memory bottleneck detected")
        
        if not recommendations:
            recommendations.append("✅ All tests performed well within acceptable parameters")
        
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
        
        print(f"\n{'='*100}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Stress test vLLM model deployed on Ray cluster")
    parser.add_argument("--ray-address", type=str, default=None, help="Ray cluster address")
    parser.add_argument("--test-type", type=str, default="concurrent", 
                        choices=["concurrent", "throughput", "latency", "memory", "all"], 
                        help="Type of stress test to run")
    parser.add_argument("--concurrent-requests", type=int, default=10, 
                        help="Number of concurrent requests for concurrent test")
    parser.add_argument("--duration", type=int, default=5, 
                        help="Test duration in minutes")
    parser.add_argument("--rps", type=int, default=5, 
                        help="Requests per second for throughput test")
    parser.add_argument("--max-tokens", type=int, default=100, 
                        help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, 
                        help="Temperature for text generation")
    parser.add_argument("--request-count", type=int, default=100, 
                        help="Number of requests for latency test")
    parser.add_argument("--memory-tokens", type=int, default=1000, 
                        help="Token count for memory stress test")
    parser.add_argument("--memory-iterations", type=int, default=20, 
                        help="Number of iterations for memory stress test")
    parser.add_argument("--output-file", type=str, default=None, 
                        help="Output file for test results")
    parser.add_argument("--verbose", action="store_true", 
                        help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize evaluator
    evaluator = StressTestEvaluator(ray_address=args.ray_address)
    
    print("="*80)
    print("RAY MODEL STRESS TEST EVALUATOR")
    print("="*80)
    
    # Connect to cluster
    if not evaluator.connect_to_cluster():
        print("❌ Failed to connect to Ray cluster")
        return 1
    
    # Check deployment
    if not evaluator.check_deployment():
        print("❌ Model deployment not found or not ready")
        return 1
    
    print(f"✅ Connected to Ray cluster")
    print(f"✅ Model deployment is available")
    print(f"🖥️  Detected {evaluator.gpu_count} GPU(s)")
    
    # Run tests
    test_results = []
    
    async def run_tests():
        nonlocal test_results
        
        if args.test_type == "concurrent" or args.test_type == "all":
            print(f"\n🚀 Running concurrent stress test...")
            result = await evaluator.concurrent_stress_test(
                concurrent_requests=args.concurrent_requests,
                duration_minutes=args.duration,
                max_tokens=args.max_tokens,
                temperature=args.temperature
            )
            test_results.append(result)
            evaluator.analyze_stress_test_results(result)
        
        if args.test_type == "throughput" or args.test_type == "all":
            print(f"\n🚀 Running throughput test...")
            result = await evaluator.throughput_test(
                requests_per_second=args.rps,
                duration_minutes=args.duration,
                max_tokens=args.max_tokens,
                temperature=args.temperature
            )
            test_results.append(result)
            evaluator.analyze_stress_test_results(result)
        
        if args.test_type == "latency" or args.test_type == "all":
            print(f"\n🚀 Running latency test...")
            result = await evaluator.latency_test(
                request_count=args.request_count,
                max_tokens=args.max_tokens,
                temperature=args.temperature
            )
            test_results.append(result)
            evaluator.analyze_stress_test_results(result)
        
        if args.test_type == "memory" or args.test_type == "all":
            print(f"\n🚀 Running memory stress test...")
            result = await evaluator.memory_stress_test(
                large_token_count=args.memory_tokens,
                iterations=args.memory_iterations
            )
            test_results.append(result)
            evaluator.analyze_stress_test_results(result)
    
    # Run async tests
    try:
        asyncio.run(run_tests())
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        evaluator.stop_continuous_monitoring()
        return 1
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        evaluator.stop_continuous_monitoring()
        return 1
    
    # Generate comprehensive report
    if len(test_results) > 1:
        evaluator.generate_performance_report(test_results)
    
    # Save results
    if args.output_file:
        if len(test_results) == 1:
            filename = evaluator.save_stress_test_results(test_results[0], args.output_file)
        else:
            # Save combined results
            combined_results = {
                'test_suite': 'comprehensive_stress_test',
                'test_timestamp': time.time(),
                'test_datetime': datetime.now().isoformat(),
                'gpu_count': evaluator.gpu_count,
                'gpu_info': evaluator.gpu_info,
                'system_info': {
                    'cpu_count': psutil.cpu_count(),
                    'total_memory_gb': psutil.virtual_memory().total / (1024**3),
                },
                'tests': test_results
            }
            with open(args.output_file, 'w') as f:
                json.dump(combined_results, f, indent=2)
            filename = args.output_file
        
        print(f"📁 Results saved to: {filename}")
    
    print(f"\n✅ Stress testing completed successfully!")
    return 0


def run_interactive_mode():
    """Interactive mode untuk memilih test secara manual"""
    evaluator = StressTestEvaluator()
    
    print("="*80)
    print("RAY MODEL STRESS TEST EVALUATOR - INTERACTIVE MODE")
    print("="*80)
    
    # Get Ray address
    ray_address = input("Enter Ray cluster address (press Enter for auto-detect): ").strip()
    if ray_address:
        evaluator.ray_address = ray_address
    
    # Connect to cluster
    if not evaluator.connect_to_cluster():
        print("❌ Failed to connect to Ray cluster")
        return
    
    # Check deployment
    if not evaluator.check_deployment():
        print("❌ Model deployment not found or not ready")
        return
    
    print(f"✅ Connected to Ray cluster")
    print(f"✅ Model deployment is available")
    print(f"🖥️  Detected {evaluator.gpu_count} GPU(s)")
    
    test_results = []
    
    while True:
        print(f"\n{'='*50}")
        print("Available Tests:")
        print("1. Concurrent Stress Test")
        print("2. Throughput Test") 
        print("3. Latency Test")
        print("4. Memory Stress Test")
        print("5. Run All Tests")
        print("6. Generate Report")
        print("7. Save Results")
        print("8. Exit")
        print(f"{'='*50}")
        
        choice = input("Select test (1-8): ").strip()
        
        if choice == "1":
            concurrent_requests = int(input("Concurrent requests [10]: ") or "10")
            duration = int(input("Duration in minutes [5]: ") or "5")
            max_tokens = int(input("Max tokens [100]: ") or "100")
            
            async def run_concurrent():
                result = await evaluator.concurrent_stress_test(
                    concurrent_requests=concurrent_requests,
                    duration_minutes=duration,
                    max_tokens=max_tokens
                )
                test_results.append(result)
                evaluator.analyze_stress_test_results(result)
            
            print("🚀 Running concurrent stress test...")
            asyncio.run(run_concurrent())
            
        elif choice == "2":
            rps = int(input("Requests per second [5]: ") or "5")
            duration = int(input("Duration in minutes [3]: ") or "3")
            max_tokens = int(input("Max tokens [50]: ") or "50")
            
            async def run_throughput():
                result = await evaluator.throughput_test(
                    requests_per_second=rps,
                    duration_minutes=duration,
                    max_tokens=max_tokens
                )
                test_results.append(result)
                evaluator.analyze_stress_test_results(result)
            
            print("🚀 Running throughput test...")
            asyncio.run(run_throughput())
            
        elif choice == "3":
            request_count = int(input("Number of requests [100]: ") or "100")
            max_tokens = int(input("Max tokens [200]: ") or "200")
            
            async def run_latency():
                result = await evaluator.latency_test(
                    request_count=request_count,
                    max_tokens=max_tokens
                )
                test_results.append(result)
                evaluator.analyze_stress_test_results(result)
            
            print("🚀 Running latency test...")
            asyncio.run(run_latency())
            
        elif choice == "4":
            token_count = int(input("Large token count [1000]: ") or "1000")
            iterations = int(input("Number of iterations [20]: ") or "20")
            
            async def run_memory():
                result = await evaluator.memory_stress_test(
                    large_token_count=token_count,
                    iterations=iterations
                )
                test_results.append(result)
                evaluator.analyze_stress_test_results(result)
            
            print("🚀 Running memory stress test...")
            asyncio.run(run_memory())
            
        elif choice == "5":
            print("🚀 Running all tests...")
            
            async def run_all():
                # Concurrent test
                result1 = await evaluator.concurrent_stress_test()
                test_results.append(result1)
                
                # Throughput test
                result2 = await evaluator.throughput_test()
                test_results.append(result2)
                
                # Latency test
                result3 = await evaluator.latency_test()
                test_results.append(result3)
                
                # Memory test
                result4 = await evaluator.memory_stress_test()
                test_results.append(result4)
                
                # Analyze each
                for result in [result1, result2, result3, result4]:
                    evaluator.analyze_stress_test_results(result)
            
            asyncio.run(run_all())
            
        elif choice == "6":
            if test_results:
                evaluator.generate_performance_report(test_results)
            else:
                print("⚠️  No test results available. Run some tests first.")
                
        elif choice == "7":
            if test_results:
                filename = input("Enter filename (press Enter for auto-generate): ").strip()
                if not filename:
                    timestamp = int(time.time())
                    filename = f"stress_test_results_{timestamp}.json"
                
                if len(test_results) == 1:
                    evaluator.save_stress_test_results(test_results[0], filename)
                else:
                    # Save combined results
                    combined_results = {
                        'test_suite': 'interactive_stress_test',
                        'test_timestamp': time.time(),
                        'test_datetime': datetime.now().isoformat(),
                        'gpu_count': evaluator.gpu_count,
                        'gpu_info': evaluator.gpu_info,
                        'tests': test_results
                    }
                    with open(filename, 'w') as f:
                        json.dump(combined_results, f, indent=2)
                    print(f"📁 Results saved to: {filename}")
            else:
                print("⚠️  No test results available. Run some tests first.")
                
        elif choice == "8":
            print("👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice. Please select 1-8.")


if __name__ == "__main__":
    import sys
    
    # Check if running in interactive mode
    if len(sys.argv) == 1 or (len(sys.argv) == 2 and sys.argv[1] == "--interactive"):
        run_interactive_mode()
    else:
        sys.exit(main())