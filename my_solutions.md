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
(a) It takes me 125.79 seconds and 0.121G memory to train BPE on tinystores. The longest token in the vocab is `' accomplishment'`, which make sense.  
(b) The step `_select_pair` function takes the most time, which is 90 seconds. Besides, `_build_word_freq` takes 30 seconds.

version 2: using heap to select best pair
(a) It takes me 42.39 seconds and 0.219G memory to train BPE on tinystores. The longest token in the vocab is `' accomplishment'`, which make sense.  
(b) The step `_build_word_freq` function takes the most time, which is 30 seconds. 





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

     

