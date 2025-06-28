import os
from ray import serve
from ray.serve.llm import LLMConfig, build_openai_app

llm_config = LLMConfig(
    model_loading_config=dict(
        model_id="qwen-2.5-7b",
        model_source="Qwen/Qwen2.5-7B-Instruct",
    ),
    deployment_config=dict(
        # Single replica pada worker node dengan GPU
        num_replicas=1,
        ray_actor_options=dict(
            num_gpus=1,  # 1 GPU dari worker node
            num_cpus=2,  # Gunakan 2 CPU cores
        ),
    ),
    # Scaling config untuk placement group
    scaling_config=dict(
        placement_group_bundles=[{"GPU": 1, "CPU": 2}],
        placement_group_strategy="STRICT_PACK",
    ),
    # Uncomment jika ingin specify accelerator type
    # accelerator_type="A10G",
    runtime_env=dict(
        env_vars=dict(
            HF_TOKEN=os.getenv("HF_TOKEN"),
            CUDA_VISIBLE_DEVICES="0",
        )
    ),
    engine_kwargs=dict(
        tensor_parallel_size=1,  # Single GPU
        max_model_len=4096,
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,  # Sedikit lebih tinggi untuk single GPU
        enforce_eager=True,
        max_num_seqs=64,  # Conservative untuk Qwen2.5-7B
        disable_log_stats=False,
    ),
)

app = build_openai_app({"llm_configs": [llm_config]})

if __name__ == "__main__":
    serve.run(app, blocking=True)