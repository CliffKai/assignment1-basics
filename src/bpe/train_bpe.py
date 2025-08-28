# src/bpe/train_bpe.py

from __future__ import annotations

import os
import regex as re
from collections import Counter
from multiprocessing import Pool, cpu_count
from typing import IO, BinaryIO, Any

# 从你的tokenizer实现中导入预分词模式
# 来源: cs336_spring2025_assignment1_basics.pdf, page 6
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def _get_stats(vocab: dict[tuple[str, ...], int]) -> Counter:
    """
    计算一个词汇表中所有相邻token对的频率。

    Args:
        vocab (dict[tuple[str, ...], int]): 一个映射，键是表示为字符串元组的单词，值是其频率。
            例如: {('l', 'o', 'w'): 5, ('n', 'e', 'w', 'e', 's', 't'): 6}

    Returns:
        Counter: 一个计数器，存储了每个相邻token对的总频率。
    """
    pairs = Counter()
    for word_tokens, freq in vocab.items():
        for i in range(len(word_tokens) - 1):
            pairs[(word_tokens[i], word_tokens[i+1])] += freq
    return pairs

def _merge_vocab(pair: tuple[str, str], v_in: dict[tuple[str, ...], int]) -> dict[tuple[str, ...], int]:
    """
    在词汇表中执行一次合并操作。

    Args:
        pair (tuple[str, str]): 需要被合并的token对，例如 ('e', 's')。
        v_in (dict[tuple[str, ...], int]): 当前的词汇表（单词 -> 频率）。

    Returns:
        dict[tuple[str, ...], int]: 合并操作后的新词汇表。
    """
    v_out = {}
    bigram = re.escape(' '.join(pair))
    # 正则表达式 p 查找不被其他字符跟随的 bigram
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    
    for word_tokens, freq in v_in.items():
        word_str = ' '.join(word_tokens)
        # 使用 p.sub 将 'e s' 替换为 'es'
        new_word_str = p.sub(''.join(pair), word_str)
        new_word_tokens = tuple(new_word_str.split(' '))
        v_out[new_word_tokens] = freq
        
    return v_out

def _process_chunk(chunk: str, special_tokens_pattern: re.Pattern | None) -> Counter:
    """
    处理单个文本块：分割特殊字符，预分词，并计算词频。
    这是由并行工作进程执行的函数。

    Args:
        chunk (str): 待处理的文本块。
        special_tokens_pattern (re.Pattern | None): 用于分割特殊字符的已编译正则表达式。

    Returns:
        Counter: 该文本块中每个预分词块的频率计数。
    """
    word_counts = Counter()
    
    # 编译主预分词模式
    tokenizer_pattern = re.compile(PAT)
    
    sub_chunks = [chunk]
    # 1. 根据特殊字符分割块
    if special_tokens_pattern:
        sub_chunks = special_tokens_pattern.split(chunk)

    for sub_chunk in sub_chunks:
        if not sub_chunk:
            continue
        # 2. 对每个子块进行预分词
        # 这里的 sub_chunk 保证不包含任何特殊字符
        pre_tokens = tokenizer_pattern.findall(sub_chunk)
        # 3. 计算频率
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

    该函数实现了BPE训练的核心逻辑，包括高效的并行预分词和优化的合并循环。

    Args:
        input_path (str | os.PathLike): 训练数据文本文件的路径。
        vocab_size (int): 最终词汇表的目标大小（包括初始字节和特殊字符）。
        special_tokens (list[str]): 一个特殊字符串列表，它们不会被分割。

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            - vocab: 训练好的分词器词汇表，一个从整数ID到其字节表示的映射。
            - merges: BPE合并规则列表，每个元素是一个字节元组，按创建顺序列出。
    """
    # --- 1. 初始化词汇表 ---
    # 初始词汇表包含所有256个字节
    vocab = {i: bytes([i]) for i in range(256)}
    
    # 将特殊字符添加到词汇表中
    for token in special_tokens:
        token_bytes = token.encode("utf-8")
        if token_bytes not in vocab.values():
            vocab[len(vocab)] = token_bytes
            
    # --- 2. 并行预分词和词频统计 ---
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    # 创建用于分割特殊字符的正则表达式
    special_tokens_pattern = None
    if special_tokens:
        # 使用 re.escape 来安全处理特殊字符中的正则表达式元字符
        escaped_tokens = [re.escape(st) for st in special_tokens]
        special_tokens_pattern = re.compile(f"({'|'.join(escaped_tokens)})")

    # 将文本分割成块以进行并行处理
    num_procs = cpu_count()
    chunk_size = (len(text) + num_procs - 1) // num_procs
    chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
    
    # 使用多进程池来并行处理块
    with Pool(num_procs) as pool:
        # functools.partial 不能被 pickle，所以使用 lambda
        results = pool.map(
            lambda chunk: _process_chunk(chunk, special_tokens_pattern), chunks
        )
    
    # 合并所有工作进程的结果
    word_counts = Counter()
    for result in results:
        word_counts.update(result)
        
    # 将单词转换为UTF-8字节，然后再转换为字符元组以进行合并
    # 例如： "apple" -> b"apple" -> ('a', 'p', 'p', 'l', 'e')
    splits = {
        tuple(word): freq
        for word, freq in word_counts.items()
    }

    # --- 3. BPE 合并循环 ---
    merges = []
    num_merges = vocab_size - len(vocab)
    
    for i in range(num_merges):
        # 计算当前所有相邻对的频率
        pair_stats = _get_stats(splits)
        
        # 如果没有更多的对可以合并，则提前停止
        if not pair_stats:
            break
            
        # 找到频率最高的对。如果频率相同，Python的max会根据字典序选择
        # 这保证了确定性
        best_pair = max(pair_stats, key=pair_stats.get)
        
        # 执行合并
        splits = _merge_vocab(best_pair, splits)
        
        # 记录这次合并
        merges.append(tuple(p.encode("utf-8") for p in best_pair))
        
        # 将新生成的token添加到词汇表中
        new_token_bytes = "".join(best_pair).encode("utf-8")
        vocab[len(vocab)] = new_token_bytes

    return vocab, merges