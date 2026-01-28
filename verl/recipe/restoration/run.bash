set -x
ENGINE=${1:-vllm}


export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=1,2,3,4
export NCCL_DEBUG=TRACE 
export HYDRA_FULL_ERROR=1

PROJECT_NAME="Opera"
EXPERIMENT_NAME="Opera"
REF_MODEL_PATH=/path/to/Visual-Quality-R1


export TOOL_BASE=http://127.0.0.1:23200/
export LLM_JUDGE_BASE=http://localhost:8002/v1
export BASE_DIR=experiments/Opera

export LOG_DIR=${BASE_DIR}/logs
export CHECKPOINT_DIR=${BASE_DIR}/checkpoints
export TENSORBOARD_DIR=${LOG_DIR}/tensorboard

mkdir -p ${BASE_DIR}
mkdir -p ${LOG_DIR}
mkdir -p ${TENSORBOARD_DIR}


python3 -m verl.trainer.main_ppo \
    data.train_files=[] \
    data.val_files=[] \
    data.train_batch_size=32 \
    data.max_prompt_length=13000 \
    data.max_response_length=800 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.0 \
    data.image_key=images \
    actor_rollout_ref.model.path=${REF_MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=8 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    reward.custom_reward_function.path="recipe/restoration/reward_function.py" \
    reward.custom_reward_function.name=compute_score \
    reward.reward_manager.name=naive \
    trainer.critic_warmup=0 \
    trainer.logger='["console","tensorboard"]' \
    trainer.val_before_train=False \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.default_local_dir=${CHECKPOINT_DIR}/ \
    +trainer.tensorboard_dir=${LOG_DIR}/tensorboard \
    trainer.total_epochs=15 2>&1 | tee ${LOG_DIR}/${EXPERIMENT_NAME}.log
