vllm serve /path/to/planning_agent_checkpoint \
  --served-model-name Qwen2.5-VL-7B-Instruct \
  --dtype bfloat16 \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.9 \
  --tensor-parallel-size 4
