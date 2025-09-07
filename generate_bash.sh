uv run python generate.py \
    --checkpoint_path=./checkpoints/tinystories_sanity_check/ckpt.pt \
    --vocab_path=./bpe_tokenizer/tinystories_vocab.json \
    --merges_path=./bpe_tokenizer/tinystories_merges.txt \
    --prompt="Once upon a time" \
    --max_new_tokens=50 \
    --temperature=0.8 \
    --top_p=0.9 \
    --device=cuda