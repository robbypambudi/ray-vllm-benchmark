import ray
from ray import serve

# Assuming you have a Ray cluster running and a deployment named 'my_deployment'
ray.init(address="auto")  # Or your cluster address
serve.start()

# Delete the deployment
serve.delete(name="vllm-model")

# Verify deletion (optional)
# from ray.serve import status
# print(status.get_app_status()) # Should not show 'vllm-model'