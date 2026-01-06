## Solutions

#### Problem(unicode1): Understanding Unicode (1 point)

(a) `chr(0)` is the Unicode character U+0000, which is called the NUL (Null) character with length=1. In python interpreter, It will return '\x00', but if we print it using `print(chr(0))`, we will get nothing visible.



(b) `repr(chr(0))` will return"'\\\x00'", this allows developers to see and reason about otherwise invisible characters. It is designed for developers and debugging.



(c) `print()`shows the contents of a string, while `repr()` shows the string as a string. `print()`or`str()`is designed for human-readable output, it shows the value of the object.

#### Problem (unicode2): Unicode Encodings (3 points)

(a) More common in real world text、more compact (shorter sequence)、less 0 bytes (wasteful).

(b) `decode_utf8_bytes_to_str_wrong("你好".encode("utf-8"))` will produce wrong output. Because one byte does not necessarily correspond to one Unicode character!

(c) My example is:

```python
bytes([228, 189]) # b'\xe4\xbd'
```

This does not decode to any Unicode character in UTF-8, because 0xE4 (1110xxxx) indicates the start of a 3-byte UTF-8 sequence, but only one continuation byte (0xBD) follows.

```python
>>> bytes([228, 189, 160]).decode("utf-8")
'你'
>>> bytes([228, 189]).decode("utf-8")
---------------------------------------------------------------------------
UnicodeDecodeError                        Traceback (most recent call last)
Cell In[50], line 1
----> 1 bytes([228, 189]).decode("utf-8")

UnicodeDecodeError: 'utf-8' codec can't decode bytes in position 0-1: unexpected end of data
```

#### Problem (train_bpe_tinystories): BPE Training on TinyStories
(a) It takes me 125.79 seconds and 0.121G memory to train BPE on tinystores. The longest token in the vocab is `'Ġaccomplishment'`, which make sense.  
(b) The step `_select_pair` function takes the most time, which is 90 seconds. Besides, `_build_word_freq` takes 30 seconds.

**version 2: Using heap to select best pair**

(a) It takes me 42.39 seconds and 0.219G memory to train BPE on tinystores. The longest token in the vocab is `'Ġaccomplishment'`, which make sense.  
(b) The step `_build_word_freq` function takes the most time, which is 30 seconds. 

#### Problem (train_bpe_expts_owt): BPE Training on OpenWebText
(a) It takes me 7569 seconds and 24.34 G memory to train BPE, the logs during training can be found at [here](logs/train_bpe_owt_log.txt).   
<table>
<tr>
<td>

![vocab size vs time](logs/train_bpe_owt_figure_speed.jpeg)

</td>
<td>

![vocab size vs pair size](logs/train_bpe_owt_figure_vocab_vs_pair.jpeg)

</td>
</tr>
</table>

The longest token in the vocab is `'ÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤ'`, and this does make sense for a byte-level BPE tokenizer.

This token is not meant to represent a readable character or word. It is a high-frequency byte sequence that appears repeatedly in the training corpus. The pattern `ÃĥÃĤ` corresponds to a specific UTF-8 byte sequence that commonly arises from misinterpreted or re-encoded UTF-8 text (often called mojibake).

(b) By running this [scripts](scripts/compare_vocabs.py), we can conclude their key differences are:
``` txt
== TinyStories ==
vocab size: 10000
max token length: 15
ASCII-only ratio: 0.9852
longest tokens:
  Ġaccomplishment
  Ġdisappointment
  Ġresponsibility
  Ġuncomfortable
  Ġcompassionate

== OpenWebText ==
vocab size: 32000
max token length: 64
ASCII-only ratio: 0.98253125
longest tokens:
  ÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤ
  ----------------------------------------------------------------
  âĢĶâĢĶâĢĶâĢĶâĢĶâĢĶâĢĶâĢĶâĢĶâĢĶâĢĶâĢĶâĢĶâĢĶâĢĶâĢĶ
  --------------------------------
  ________________________________

== Overlap ==
shared tokens: 7319
only TinyStories: 2681
only OpenWebText: 24681
```
 - The TinyStories tokenizer learns mostly short, human-readable English word and subword tokens, reflecting the clean, simple, and homogeneous nature of the corpus. 
 - The OpenWebText tokenizer learns a much larger and more diverse vocabulary, including very long byte-level tokens and noisy UTF-8 artifacts, capturing the heterogeneous, messy, and web-scale characteristics of internet text.

#### Problem (tokenizer_experiments): Experiments with tokenizers
Below are the outputs produced by from this [scripts](scripts/tokenizer_experiments.py).
```
================ problem (a) ================
TS data + TS tok: total_bytes: 9170, total_tokens: 2225 => compression_ratio: 4.1213 bytes/token
OWT data + OWT tok: total_bytes: 23892, total_tokens: 5335 => compression_ratio: 4.4784 bytes/token
================ problem (b) ================
OWT data + TS tok: total_bytes: 23892, total_tokens: 7577 => compression_ratio: 3.1532 bytes/token
TS data + OWT tok total_bytes: 9170, total_tokens: 2286 => compression_ratio: 4.0114 bytes/token
================ problem (c) ================
OWT tokenizer throughput: 3.91 MB/s
Estimated time for 825GB: 58.60 hours (2.44 days)
(optional) tokens/sec: 0.90 M tokens/s
```
(a) On 10 sampled documents, the TinyStories tokenizer achieves 4.1213 bytes/token on TinyStories and the OpenWebText tokenizer achieves 4.4784 bytes/token on OpenWebText.

(b) Tokenizing the OpenWebText sample with the TinyStories tokenizer yields a substantially worse compression ratio (3.1532 bytes/token vs 4.4784 bytes/token), indicating the smaller, domain-specific TinyStories vocabulary splits OpenWebText text into many more (shorter) tokens.

(c) The OpenWebText tokenizer runs at about 3.91 MB/s, so tokenizing the 825GB Pile would take approximately 58.6 hours (~2.44 days).

(d) uint16 is appropriate because it can represent token IDs in the range 0–65535, which safely covers vocabularies of 10K and 32K, while halving storage and I/O cost relative to int32.

## Notes

1. Unicode code point 定义“是什么字符”，

   Unicode encoding 定义“这个字符如何变成字节”。

2. 在 Python 中，bytes() 和 encode() 的关系可以用一句话概括：

   - encode() 是 str → bytes 的“专用接口”，语义明确、只做一件事、是str的方法

   - bytes() 是更通用、更底层的构造器、语义多态，可以：

     - 从 int 创建缓冲区
     - 从 iterable[int] 创建字节
     - 从 bytearray 冻结
     - 从 str + encoding 编码

     

