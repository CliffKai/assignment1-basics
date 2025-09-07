# 在 TinyStories 上从头开始训练
uv run python train.py \
    --train_data_path=/root/data/cs336/tinystories_train.npy \
    --val_data_path=/root/data/cs336/tinystories_val.npy \
    --vocab_size=10000 \
    --context_length=256 \
    --d_model=512 \
    --num_layers=4 \
    --num_heads=16 \
    --d_ff=1344 \
    --batch_size=32 \
    --max_steps=100 \
    --eval_interval=50 \
    --out_dir=./checkpoints/tinystories_sanity_check \
    --device=cuda

# 从检查点继续训练
uv run python train.py --init_from=resume