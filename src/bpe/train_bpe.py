# src/bpe/train_bpe.py

from __future__ import annotations

import os
import regex as re
from collections import Counter
from multiprocessing import Pool, cpu_count
from typing import IO, BinaryIO, Any
from itertools import repeat # 确保导入 repeat

# 从作业PDF中获取的预分词模式
# 来源: cs336_spring2025_assignment1_basics.pdf, page 6
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def _get_stats(vocab: dict[tuple[str, ...], int]) -> Counter:
    """
    计算一个词汇表中所有相邻token对的频率。
    """
    pairs = Counter()
    for word_tokens, freq in vocab.items():
        for i in range(len(word_tokens) - 1):
            pairs[(word_tokens[i], word_tokens[i+1])] += freq
    return pairs

def _merge_vocab(pair: tuple[str, str], v_in: dict[tuple[str, ...], int]) -> dict[tuple[str, ...], int]:
    """
    在词汇表中执行一次合并操作。
    """
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    
    for word_tokens, freq in v_in.items():
        word_str = ' '.join(word_tokens)
        new_word_str = p.sub(''.join(pair), word_str)
        new_word_tokens = tuple(new_word_str.split(' '))
        v_out[new_word_tokens] = freq
        
    return v_out

def _process_chunk(chunk: str, special_tokens_pattern: re.Pattern | None) -> Counter:
    """
    处理单个文本块：分割特殊字符，预分词，并计算词频。
    这是由并行工作进程执行的顶级函数，可以被pickle。
    """
    word_counts = Counter()
    tokenizer_pattern = re.compile(PAT)
    
    sub_chunks = [chunk]
    if special_tokens_pattern:
        sub_chunks = special_tokens_pattern.split(chunk)

    for sub_chunk in sub_chunks:
        if not sub_chunk:
            continue
        pre_tokens = tokenizer_pattern.findall(sub_chunk)
        word_counts.update(pre_tokens)
        
    return word_counts


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs: Any,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    根据给定的输入语料库训练一个字节对编码（BPE）分词器。
    """
    # 1. 初始化词汇表
    vocab = {i: bytes([i]) for i in range(256)}
    for token in special_tokens:
        token_bytes = token.encode("utf-8")
        if token_bytes not in vocab.values():
            vocab[len(vocab)] = token_bytes
            
    # 2. 并行预分词和词频统计
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    special_tokens_pattern = None
    if special_tokens:
        escaped_tokens = [re.escape(st) for st in special_tokens]
        special_tokens_pattern = re.compile(f"({'|'.join(escaped_tokens)})")

    num_procs = cpu_count()
    chunk_size = (len(text) + num_procs - 1) // num_procs
    chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
    
    # 使用 starmap 替代 map，避免使用不可序列化的 lambda 函数
    starmap_args = zip(chunks, repeat(special_tokens_pattern))
    
    with Pool(num_procs) as pool:
        results = pool.starmap(_process_chunk, starmap_args)
    
    word_counts = Counter()
    for result in results:
        word_counts.update(result)
        
    # 将单词字符串转换为字符元组以进行合并
    splits = { tuple(word): freq for word, freq in word_counts.items() }

    # 3. BPE 合并循环
    merges = []
    num_merges = vocab_size - len(vocab)
    
    for i in range(num_merges):
        pair_stats = _get_stats(splits)
        if not pair_stats:
            break
            
        best_pair = max(pair_stats, key=pair_stats.get)
        
        splits = _merge_vocab(best_pair, splits)
        
        merges.append(tuple(p.encode("utf-8") for p in best_pair))
        
        new_token_bytes = "".join(best_pair).encode("utf-8")
        vocab[len(vocab)] = new_token_bytes

    return vocab, merges