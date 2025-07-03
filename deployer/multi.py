#!/usr/bin/env python3
"""
Qwen Model Deployment untuk Ray dengan Pipeline Parallelism
Equivalent dari: vllm serve qwen --tensor-parallel-size 4 --pipeline-parallel-size 2
"""

import os
import time
import logging
import argparse
import json
from typing import Dict, Any

import ray
from ray import serve

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def apply_vllm_cuda_fix():
    """
    Solusi komprehensif untuk CUDA issues di Ray + vLLM
    Fixes: CUDA_VISIBLE_DEVICES dan device_type empty string errors
    """
    logger.info("Applying comprehensive vLLM CUDA environment fix...")
    
    # Log current state
    current_cuda = os.environ.get("CUDA_VISIBLE_DEVICES", "NOT_SET")
    current_device = os.environ.get("CUDA_DEVICE_ORDER", "NOT_SET")
    logger.info(f"Before fix - CUDA_VISIBLE_DEVICES: '{current_cuda}'")
    logger.info(f"Before fix - CUDA_DEVICE_ORDER: '{current_device}'")
    
    # THE CORRECT SOLUTION: Delete CUDA_VISIBLE_DEVICES
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        del os.environ["CUDA_VISIBLE_DEVICES"]
        logger.info("✓ Deleted CUDA_VISIBLE_DEVICES - vLLM will auto-detect GPUs")
    
    # Remove Ray CUDA conflicts
    conflicting_vars = [
        "RAY_CUDA_VISIBLE_DEVICES",
        "RAY_GPU_IDS", 
        "RAY_OVERRIDE_CUDA_VISIBLE_DEVICES"
    ]
    
    for var in conflicting_vars:
        if var in os.environ:
            del os.environ[var]
            logger.info(f"✓ Removed {var}")
    
    # CRITICAL: Set device-related environment variables to prevent empty device_type
    # This fixes the "Device string must not be empty" error
    device_env_vars = {
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "VLLM_TARGET_DEVICE": "cuda",
        "CUDA_LAUNCH_BLOCKING": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "VLLM_ATTENTION_BACKEND": "FLASH_ATTN",
        "VLLM_USE_TRITON_FLASH_ATTN": "True",
        # Additional fixes for device detection
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128",
        "VLLM_LOGGING_LEVEL": "INFO"
    }
    
    for var, value in device_env_vars.items():
        os.environ[var] = value
        logger.info(f"✓ Set {var}={value}")
    
    # Verify CUDA availability
    try:
        import torch
        logger.info(f"PyTorch CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"PyTorch CUDA device count: {torch.cuda.device_count()}")
            logger.info(f"PyTorch current device: {torch.cuda.current_device()}")
        else:
            logger.warning("PyTorch reports CUDA as unavailable!")
    except ImportError:
        logger.warning("PyTorch not available for verification")
    
    # Get Ray GPU information
    try:
        import ray
        if ray.is_initialized():
            gpu_ids = ray.get_gpu_ids()
            logger.info(f"Ray GPU IDs: {gpu_ids}")
        else:
            logger.info("Ray not initialized yet")
    except Exception as e:
        logger.info(f"Could not get Ray GPU info: {e}")
    
    final_cuda = os.environ.get("CUDA_VISIBLE_DEVICES", "NOT_SET")
    logger.info(f"After fix - CUDA_VISIBLE_DEVICES: '{final_cuda}'")
    logger.info("✓ Comprehensive CUDA fix completed")

# Configuration for different Qwen models
QWEN_MODEL_CONFIGS = {
    "qwen2.5-0.5b": {
        "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
        "recommended_tp": 1,
        "recommended_pp": 1,
        "max_model_len": 4096,
        "min_gpus": 1
    },
    "qwen2.5-1.5b": {
        "model_name": "Qwen/Qwen2.5-1.5B-Instruct", 
        "recommended_tp": 2,
        "recommended_pp": 1,
        "max_model_len": 8192,
        "min_gpus": 2
    },
    "qwen2.5-3b": {
        "model_name": "Qwen/Qwen2.5-3B-Instruct",
        "recommended_tp": 2,
        "recommended_pp": 2,
        "max_model_len": 8192,
        "min_gpus": 4
    },
    "qwen2.5-7b": {
        "model_name": "Qwen/Qwen2.5-7B-Instruct",
        "recommended_tp": 4,
        "recommended_pp": 2,
        "max_model_len": 8192,
        "min_gpus": 8
    },
    "qwen2.5-14b": {
        "model_name": "Qwen/Qwen2.5-14B-Instruct",
        "recommended_tp": 4,
        "recommended_pp": 4,
        "max_model_len": 8192,
        "min_gpus": 16
    },
    "qwen2.5-32b": {
        "model_name": "Qwen/Qwen2.5-32B-Instruct",
        "recommended_tp": 8,
        "recommended_pp": 4,
        "max_model_len": 8192,
        "min_gpus": 32
    },
    "qwen2.5-coder-7b": {
        "model_name": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "recommended_tp": 4,
        "recommended_pp": 2,
        "max_model_len": 8192,
        "min_gpus": 8
    }
}

def create_qwen_deployment(model_config: Dict[str, Any], tensor_parallel_size: int, pipeline_parallel_size: int):
    """
    Create Qwen deployment equivalent to vllm serve command
    """
    
    total_gpus = tensor_parallel_size * pipeline_parallel_size
    model_name = model_config["model_name"]
    max_model_len = model_config["max_model_len"]
    
    # Calculate placement group bundles for distributed deployment
    bundles = []
    
    # Driver bundle (no GPU needed)
    bundles.append({"CPU": 2})
    
    # Worker bundles for each pipeline stage
    for pp_stage in range(pipeline_parallel_size):
        for tp_rank in range(tensor_parallel_size):
            bundles.append({"CPU": 2, "GPU": 1})
    
    logger.info(f"Creating Qwen deployment:")
    logger.info(f"  Model: {model_name}")
    logger.info(f"  Tensor Parallel: {tensor_parallel_size}")
    logger.info(f"  Pipeline Parallel: {pipeline_parallel_size}")
    logger.info(f"  Total GPUs needed: {total_gpus}")
    logger.info(f"  Placement bundles: {len(bundles)}")
    
    @serve.deployment(
        name="qwen-vllm-serve",
        num_replicas=1,
        placement_group_bundles=bundles,
        placement_group_strategy="STRICT_SPREAD" if pipeline_parallel_size > 1 else "STRICT_PACK"
    )
    class QwenVLLMServeDeployment:
        """
        Qwen vLLM Deployment - Ray equivalent of 'vllm serve' command
        """
        
        def __init__(self):
            logger.info(f"Initializing Qwen vLLM Serve deployment...")
            logger.info(f"Equivalent to: vllm serve {model_name} --tensor-parallel-size {tensor_parallel_size} --pipeline-parallel-size {pipeline_parallel_size}")
            
            # Apply CUDA fix FIRST - critical for Ray deployment
            apply_vllm_cuda_fix()
            
            # Import vLLM after CUDA fix
            from vllm import LLM, SamplingParams
            
            logger.info("Initializing vLLM with distributed configuration...")
            
            # vLLM configuration equivalent to vllm serve command
            vllm_config = {
                "model": model_name,
                "tensor_parallel_size": tensor_parallel_size,
                "pipeline_parallel_size": pipeline_parallel_size,
                "max_model_len": max_model_len,
                "trust_remote_code": True,
                "enforce_eager": True,  # For stability in distributed setup
                "gpu_memory_utilization": 0.85,
                "disable_log_stats": False,
                "distributed_executor_backend": "ray",  # Use Ray for distributed execution
                # "worker_use_ray": True,
                "max_parallel_loading_workers": min(total_gpus, 8),
                # Additional Qwen-specific optimizations
                "dtype": "auto",
                "load_format": "auto",
                "quantization": None,
                "seed": 0,
                "max_num_batched_tokens": None,
                "max_num_seqs": 256,
                "disable_sliding_window": False,
                # CRITICAL: Explicit device configuration to prevent empty device_type
                "device": "cuda",  # Explicitly set device type
                # Additional device-related configs
                "disable_custom_all_reduce": False,
                "disable_log_requests": False,
            }
            
            # Initialize vLLM
            self.llm = LLM(**vllm_config)
            
            # Store configuration
            self.model_name = model_name
            self.tensor_parallel_size = tensor_parallel_size
            self.pipeline_parallel_size = pipeline_parallel_size
            self.total_gpus = total_gpus
            
            logger.info("✓ Qwen vLLM deployment initialized successfully!")
            logger.info(f"✓ Model loaded with TP={tensor_parallel_size}, PP={pipeline_parallel_size}")
        
        async def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7, top_p: float = 0.9):
            """
            Generate text using Qwen model - equivalent to vLLM serve API
            """
            try:
                from vllm import SamplingParams
                
                # Qwen-optimized sampling parameters
                sampling_params = SamplingParams(
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    stop=["<|endoftext|>", "<|im_end|>", "</s>"]
                )
                
                # Generate
                outputs = self.llm.generate([prompt], sampling_params)
                
                if outputs and len(outputs) > 0:
                    output = outputs[0]
                    generated_text = output.outputs[0].text
                    
                    # Estimate token counts (rough approximation)
                    prompt_tokens = len(prompt.split()) * 1.3
                    completion_tokens = len(generated_text.split()) * 1.3
                    
                    return {
                        "text": generated_text,
                        "prompt_tokens": int(prompt_tokens),
                        "completion_tokens": int(completion_tokens), 
                        "total_tokens": int(prompt_tokens + completion_tokens),
                        "model": self.model_name,
                        "tensor_parallel_size": self.tensor_parallel_size,
                        "pipeline_parallel_size": self.pipeline_parallel_size
                    }
                
                return {"error": "No output generated"}
                
            except Exception as e:
                logger.error(f"Generation error: {e}")
                return {"error": str(e)}
        
        async def chat_completions(self, messages: list, max_tokens: int = 512, temperature: float = 0.7):
            """
            OpenAI-compatible chat completions endpoint for Qwen
            """
            try:
                # Convert messages to Qwen format
                if isinstance(messages, list) and len(messages) > 0:
                    # Simple message formatting for Qwen
                    if messages[-1].get("role") == "user":
                        prompt = f"<|im_start|>user\n{messages[-1]['content']}<|im_end|>\n<|im_start|>assistant\n"
                    else:
                        prompt = str(messages[-1].get("content", ""))
                else:
                    prompt = str(messages)
                
                # Generate response
                result = await self.generate(prompt, max_tokens, temperature)
                
                # Format as OpenAI-compatible response
                if "error" not in result:
                    return {
                        "choices": [{
                            "message": {
                                "role": "assistant",
                                "content": result["text"]
                            },
                            "finish_reason": "stop"
                        }],
                        "usage": {
                            "prompt_tokens": result.get("prompt_tokens", 0),
                            "completion_tokens": result.get("completion_tokens", 0),
                            "total_tokens": result.get("total_tokens", 0)
                        },
                        "model": self.model_name
                    }
                else:
                    return {"error": result["error"]}
                    
            except Exception as e:
                logger.error(f"Chat completion error: {e}")
                return {"error": str(e)}
        
        async def health_check(self):
            """Health check endpoint"""
            return {
                "status": "healthy",
                "model": self.model_name,
                "tensor_parallel_size": self.tensor_parallel_size,
                "pipeline_parallel_size": self.pipeline_parallel_size,
                "total_gpus": self.total_gpus,
                "timestamp": time.time(),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "NOT_SET"),
                "deployment_type": "qwen_vllm_serve_equivalent"
            }
        
        async def model_info(self):
            """Get detailed model information"""
            return {
                "model_name": self.model_name,
                "tensor_parallel_size": self.tensor_parallel_size,
                "pipeline_parallel_size": self.pipeline_parallel_size,
                "total_workers": self.total_gpus,
                "max_model_len": max_model_len,
                "deployment_equivalent": f"vllm serve {model_name} --tensor-parallel-size {tensor_parallel_size} --pipeline-parallel-size {pipeline_parallel_size}",
                "distributed_executor": "ray"
            }
    
    return QwenVLLMServeDeployment

class QwenRayDeployer:
    """Deployer for Qwen models using Ray - equivalent to vllm serve"""
    
    def __init__(self, ray_address=None):
        self.ray_address = ray_address
        self.connected = False
    
    def connect_to_cluster(self):
        """Connect to Ray cluster"""
        logger.info(f"Connecting to Ray cluster: {self.ray_address or 'auto-detect'}")
        
        try:
            if self.ray_address:
                ray.init(address=self.ray_address, ignore_reinit_error=True)
            else:
                ray.init(address='auto', ignore_reinit_error=True)
            
            # Check cluster resources
            resources = ray.cluster_resources()
            logger.info(f"Cluster resources: {resources}")
            
            self.connected = True
            logger.info("✓ Connected to Ray cluster")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            return False
    
    def validate_deployment_requirements(self, model_key: str, tensor_parallel_size: int, pipeline_parallel_size: int):
        """Validate cluster can support the deployment"""
        if model_key not in QWEN_MODEL_CONFIGS:
            logger.error(f"Unknown model: {model_key}")
            logger.info(f"Available models: {list(QWEN_MODEL_CONFIGS.keys())}")
            return False
        
        config = QWEN_MODEL_CONFIGS[model_key]
        total_gpus_needed = tensor_parallel_size * pipeline_parallel_size
        
        try:
            cluster_resources = ray.cluster_resources()
            available_gpus = cluster_resources.get('GPU', 0)
            
            logger.info(f"Deployment requirements:")
            logger.info(f"  Model: {config['model_name']}")
            logger.info(f"  Recommended TP: {config['recommended_tp']}")
            logger.info(f"  Recommended PP: {config['recommended_pp']}")
            logger.info(f"  Minimum GPUs: {config['min_gpus']}")
            logger.info(f"  Requested TP×PP: {tensor_parallel_size}×{pipeline_parallel_size} = {total_gpus_needed} GPUs")
            logger.info(f"  Available GPUs: {available_gpus}")
            
            if available_gpus < total_gpus_needed:
                logger.error(f"Insufficient GPUs! Need {total_gpus_needed}, have {available_gpus}")
                return False
            
            # Check nodes for pipeline parallelism
            if pipeline_parallel_size > 1:
                nodes = ray.nodes()
                gpu_nodes = [n for n in nodes if n.get('Resources', {}).get('GPU', 0) > 0 and n.get('Alive', False)]
                
                if len(gpu_nodes) < pipeline_parallel_size:
                    logger.error(f"Pipeline parallelism needs {pipeline_parallel_size} GPU nodes, have {len(gpu_nodes)}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False
    
    def deploy_qwen_model(self, model_key: str, tensor_parallel_size: int = None, pipeline_parallel_size: int = None):
        """
        Deploy Qwen model equivalent to: vllm serve qwen --tensor-parallel-size X --pipeline-parallel-size Y
        """
        if not self.connected:
            logger.error("Not connected to Ray cluster")
            return False
        
        if model_key not in QWEN_MODEL_CONFIGS:
            logger.error(f"Unknown model: {model_key}")
            return False
        
        config = QWEN_MODEL_CONFIGS[model_key]
        
        # Use recommended values if not specified
        if tensor_parallel_size is None:
            tensor_parallel_size = config["recommended_tp"]
        if pipeline_parallel_size is None:
            pipeline_parallel_size = config["recommended_pp"]
        
        # Validate requirements
        if not self.validate_deployment_requirements(model_key, tensor_parallel_size, pipeline_parallel_size):
            return False
        
        logger.info(f"Deploying Qwen model equivalent to:")
        logger.info(f"vllm serve {config['model_name']} --tensor-parallel-size {tensor_parallel_size} --pipeline-parallel-size {pipeline_parallel_size}")
        
        try:
            # Create deployment
            deployment_class = create_qwen_deployment(config, tensor_parallel_size, pipeline_parallel_size)
            deployment = deployment_class.bind()
            
            # Deploy
            serve.run(deployment, name="qwen-vllm-serve", route_prefix="/")
            
            # Wait for deployment
            if self._wait_for_deployment():
                logger.info("✓ Qwen model deployed successfully!")
                self._save_deployment_info(model_key, config, tensor_parallel_size, pipeline_parallel_size)
                return True
            else:
                logger.error("✗ Deployment failed")
                return False
                
        except Exception as e:
            logger.error(f"Deployment error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _wait_for_deployment(self, timeout=600):
        """Wait for deployment to be ready"""
        logger.info("Waiting for Qwen deployment...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                status_info = serve.status()
                if hasattr(status_info, "applications") and "qwen-vllm-serve" in status_info.applications:
                    app_status = status_info.applications["qwen-vllm-serve"].status.value
                    logger.info(f"Deployment status: {app_status}")
                    
                    if app_status == "RUNNING":
                        return True
                    elif app_status == "DEPLOY_FAILED":
                        return False
                        
            except Exception as e:
                logger.debug(f"Waiting: {e}")
            
            print(".", end="", flush=True)
            time.sleep(10)
        
        print()
        return False
    
    def _save_deployment_info(self, model_key: str, config: Dict, tp: int, pp: int):
        """Save deployment information"""
        info = {
            "model_key": model_key,
            "model_name": config["model_name"],
            "tensor_parallel_size": tp,
            "pipeline_parallel_size": pp,
            "total_gpus": tp * pp,
            "equivalent_command": f"vllm serve {config['model_name']} --tensor-parallel-size {tp} --pipeline-parallel-size {pp}",
            "deployment_time": time.time(),
            "status": "deployed"
        }
        
        with open("qwen_deployment_info.json", "w") as f:
            json.dump(info, f, indent=2)
        
        logger.info("Deployment info saved to qwen_deployment_info.json")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Deploy Qwen models with Ray - equivalent to vllm serve")
    parser.add_argument("--ray-address", type=str, default=None, help="Ray cluster address")
    parser.add_argument("--model", type=str, required=True, 
                        choices=list(QWEN_MODEL_CONFIGS.keys()),
                        help="Qwen model to deploy")
    parser.add_argument("--tensor-parallel-size", type=int, default=None,
                        help="Tensor parallel size (default: model recommended)")
    parser.add_argument("--pipeline-parallel-size", type=int, default=None,
                        help="Pipeline parallel size (default: model recommended)")
    parser.add_argument("--list-models", action="store_true", 
                        help="List available Qwen models")
    parser.add_argument("--action", type=str, choices=["deploy", "status", "stop"], 
                        default="deploy", help="Action to perform")
    
    args = parser.parse_args()
    
    if args.list_models:
        print("Available Qwen models:")
        for key, config in QWEN_MODEL_CONFIGS.items():
            print(f"  {key}:")
            print(f"    Model: {config['model_name']}")
            print(f"    Recommended TP: {config['recommended_tp']}")
            print(f"    Recommended PP: {config['recommended_pp']}")
            print(f"    Min GPUs: {config['min_gpus']}")
            print()
        return
    
    print("Qwen Ray Deployment - vLLM Serve Equivalent")
    print("=" * 50)
    
    deployer = QwenRayDeployer(args.ray_address)
    
    try:
        if not deployer.connect_to_cluster():
            print("Failed to connect to Ray cluster")
            return
        
        if args.action == "deploy":
            config = QWEN_MODEL_CONFIGS[args.model]
            tp = args.tensor_parallel_size or config["recommended_tp"]
            pp = args.pipeline_parallel_size or config["recommended_pp"]
            
            print(f"\nDeploying Qwen model:")
            print(f"  Model: {config['model_name']}")
            print(f"  Tensor Parallel: {tp}")
            print(f"  Pipeline Parallel: {pp}")
            print(f"  Equivalent: vllm serve {config['model_name']} --tensor-parallel-size {tp} --pipeline-parallel-size {pp}")
            
            if deployer.deploy_qwen_model(args.model, tp, pp):
                print("✓ Qwen model deployed successfully!")
                print("Access endpoints:")
                print("  - Generate: POST /generate")
                print("  - Chat: POST /chat_completions") 
                print("  - Health: GET /health_check")
            else:
                print("✗ Deployment failed")
        
        elif args.action == "status":
            try:
                status = serve.status()
                print(f"Deployment status: {status}")
            except Exception as e:
                print(f"Error checking status: {e}")
        
        elif args.action == "stop":
            try:
                serve.shutdown()
                print("✓ Deployment stopped")
            except Exception as e:
                print(f"Error stopping: {e}")
    
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            ray.disconnect()
        except:
            pass

if __name__ == "__main__":
    main()