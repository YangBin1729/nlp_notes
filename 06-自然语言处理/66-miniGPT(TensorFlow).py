# -*- coding: utf-8 -*-
# 原始链接：https://keras.io/examples/generative/text_generation_with_miniature_gpt/

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import numpy as np
import os
import re
import string
import random


# # 自注意力层
# > TODO：填充的掩码，如何处理？

class MultiHeadAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = "
                f"{num_heads}")
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combined_heads = layers.Dense(embed_dim)

    @staticmethod
    def casual_attention_mask(n_dest, n_src, dtype):
        """
        n_dest： 目标序列长度
        n_src： 源序列长度
        return： [n_dest,n_src]
        """

        i = tf.range(n_dest)[:, None]
        j = tf.range(n_src)
        m = i >= j - n_src + n_dest
        return tf.cast(m, dtype)

    def attention(self, query, key, value):
        """
        query/key/value: (batch_size, num_heads, seq_len, projection_dim)

        """

        # (batch_size, num_heads, seq_len, seq_len)
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)

        # 防止获取到 当前标记 后面标记的信息
        shape = tf.shape(scaled_score)
        dim_dest, dim_src = shape[2], shape[3]
        attention_mask = self.casual_attention_mask(
            dim_dest,
            dim_src,
            scaled_score.dtype,
        )
        attention_mask = tf.reshape(attention_mask, [1, 1, dim_dest, dim_src])
        scaled_score = scaled_score * attention_mask - 1e4 * (1 -
                                                              attention_mask)

        # (batch_size, num_heads, seq_len, seq_len)
        weights = tf.nn.softmax(scaled_score, axis=-1)

        # (batch_size, num_heads, seq_len, projection_dim)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(
            x,
            (batch_size, -1, self.num_heads, self.projection_dim),
        )
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs, **kwargs):
        # batch_size, seq_len, embedding_size
        batch_size = tf.shape(inputs)[0]

        # (batch_size, seq_len, embed_dim)
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)

        # (batch_size, num_heads, seq_len, projection_dim)
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)

        # (batch_size, num_heads, seq_len, projection_dim)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention,
                                      (batch_size, -1, self.embed_dim))
        # (batch_size, seq_len, embed_size)
        output = self.combined_heads(concat_attention)
        return output


# +
# 创建掩码，遮掩 当前标记 之后的所有标记
n_dest = 4
n_src = 4


def create_attention_mask(n_dest, n_src, dtype=tf.float32):
    i = tf.range(n_dest)[:, None]
    j = tf.range(n_src)
    m = i >= j - n_src + n_dest
    return tf.cast(m, dtype)


create_attention_mask(n_dest, n_src)

# +
# 掩码操作
# batch_size=1,num_heads=1, seq_len=3
scaled_score = tf.random.uniform((1, 1, 3, 3), 0, 1, dtype=tf.float32)
print("Before mask:\n", scaled_score)

_, _, dim_dest, dim_src = tf.shape(scaled_score)
attention_mask = create_attention_mask(dim_dest, dim_src)

scaled_score = scaled_score * attention_mask - 1e4 * (1 - attention_mask)
print("After mask:\n", scaled_score)
# -

# batch_size=1, seq_len=3, embed_size=4
inputs = tf.random.uniform((1, 3, 4), 0, 1, dtype=tf.float32)
embed_dim = 10
num_heads = 5
attention = MultiHeadAttention(embed_dim, num_heads)
output = attention(inputs)
print(output)


# # Transformer 层

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        
        # 前向层
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dense(embed_dim),
        ])
        
        # 正则化层
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs):  # batch_size,seq_len,embed_size
        attention_output = self.attn(inputs)  # batch_size,seq_len,embed_size
        attention_output = self.dropout1(attention_output)
        out1 = self.layernorm1(attention_output)
        ffn_output = self.ffn(out1)  # batch_size,seq_len,embed_size
        ffn_output = self.dropout2(ffn_output)  # batch_size,seq_len,embed_size
        return self.layernorm2(out1 + ffn_output)


ff_dim = 10
transformer = TransformerBlock(embed_dim, num_heads, ff_dim)
print(transformer(inputs))


# # 嵌入层

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        # 词嵌入
        self.token_embed = layers.Embedding(
            input_dim=vocab_size,
            output_dim=embed_dim,
        )
        # 位置编码,也是待训练参数
        self.pos_embed = layers.Embedding(
            input_dim=maxlen,
            output_dim=embed_dim,
        )

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_embed(positions)
        x = self.token_embed(x)
        return x + positions


# +
maxlen = 5
vocab_size = 20
embed = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)

# batch=2, seq_len=5
x = tf.constant(np.random.randint(1, 15, (2, maxlen), dtype=np.int32))
print(embed(x))
# -


# # GPT 模型

vocab_size = 20000
maxlen = 100
embed_dim = 256
num_heads = 2
ff_dim = 256


def create_model():
    # batch,seq_len --> batch,seq_len,embed_size
    inputs = layers.Input(shape=(maxlen,), dtype=tf.int32)
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    x = embedding_layer(inputs)
    
    # batch,seq_len,embed_size --> batch,seq_len,vocab_size
    transformer = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer(x)
    outputs = layers.Dense(vocab_size)(x)
    model = keras.Model(inputs=inputs, outputs=[outputs, x])
    
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    model.compile("adam", loss=[loss_fn, None])
    return model


# # 训练数据

batch_size = 32
filenames = []
directories = [
    "aclImdb/train/pos",
    "aclImdb/train/neg",
    "aclImdb/test/pos",
    "aclImdb/test/neg",
]
base_dir = "../datasets"

# +
for dir in directories:
    dir = os.path.join(base_dir, dir)
    for f in os.listdir(dir):
        filenames.append(os.path.join(dir, f))

print(f"{len(filenames)} files")

# +
# 创建数据管道
random.shuffle(filenames)
text_ds = tf.data.TextLineDataset(filenames)
text_ds = text_ds.shuffle(buffer_size=256)
text_ds = text_ds.batch(batch_size)

for data in text_ds.take(1):
    tf.print(data)


# -

# 数据预处理
def custom_standardization(input_string):
    """ Remove html line-break tags and handle punctuation """

    lowercased = tf.strings.lower(input_string)
    stripped_html = tf.strings.regex_replace(lowercased, "<br />", " ")
    return tf.strings.regex_replace(stripped_html, f"([{string.punctuation}])", r" \1")


# +
# 文本数据向量化
vectorize_layer = TextVectorization(
    standardize=custom_standardization,
    max_tokens=vocab_size - 1,
    output_mode="int",
    output_sequence_length=maxlen + 1,
)
vectorize_layer.adapt(text_ds)


# 词汇表，列表的元素类型为 bytes
vocab = vectorize_layer.get_vocabulary()  
# -

for text in text_ds.take(1):
    text = tf.expand_dims(text, -1)    
    tokenized_sentences = vectorize_layer(text)
    print(tokenized_sentences)
    print([vocab[idx] for idx in tokenized_sentences[0]])





# +
# 建训练数据转换成 输入和标签
def prepare_lm_inputs_labels(text):
    text = tf.expand_dims(text, -1)
    tokenized_sentences = vectorize_layer(text)
    x = tokenized_sentences[:, :-1] 
    y = tokenized_sentences[:, 1:] # 标签相对于输入，后移一位
    return x, y


text_ds = text_ds.map(prepare_lm_inputs_labels)
text_ds = text_ds.prefetch(tf.data.experimental.AUTOTUNE)

for data in text_ds.take(1):
    print(data)


# -

# # 文本生成
# 以回调函数的形式实现

# 文本生成回调函数
class TextGenerator(keras.callbacks.Callback):
    def __init__(self,
                 max_tokens,
                 start_tokens,
                 index_to_word,
                 top_k=10,
                 print_every=1):
        self.max_tokens = max_tokens
        self.start_tokens = start_tokens
        self.index_to_word = index_to_word
        self.print_every = print_every
        self.k = top_k

    def sample_from(self, logits):
        logits, indices = tf.math.top_k(logits, k=self.k, sorted=True)
        indices = np.asarray(indices).astype("int32")
        preds = keras.activations.softmax(tf.expand_dims(logits, 0))[0]
        preds = np.asarray(preds).astype("float32")
        return np.random.choice(indices, p=preds)

    def detokenize(self, number):
        return self.index_to_word[number]

    def on_epoch_end(self, epoch, logs=None):
        start_tokens = [_ for _ in self.start_tokens]
        if (epoch + 1) % self.print_every != 0:
            return
        num_tokens_generated = 0
        tokens_generated = []
        while num_tokens_generated <= self.max_tokens:
            pad_len = maxlen - len(start_tokens)
            sample_index = len(start_tokens) - 1
            if pad_len < 0:
                x = start_tokens[:maxlen]
                sample_index = maxlen - 1
            elif pad_len > 0:
                x = start_tokens + [0] * pad_len
            else:
                x = start_tokens
            x = np.array([x])
            y, _ = self.model.predict(x)
            sample_token = self.sample_from(y[0][sample_index])
            tokens_generated.append(sample_token)
            start_tokens.append(sample_token)
            num_tokens_generated = len(tokens_generated)
        txt = " ".join(
            [self.detokenize(_) for _ in self.start_tokens + tokens_generated])
        print(f"generated text:\n{txt}\n")

    def sample_from(self, logits):
        logits, indices = tf.math.top_k(logits, k=self.k, sorted=True)
        indices = np.asarray(indices).astype("int32")
        preds = keras.activations.softmax(tf.expand_dims(logits, 0))[0]
        preds = np.asarray(preds).astype("float32")
        return np.random.choice(indices, p=preds)

    def detokenize(self, number):
        return self.index_to_word[number]

    def on_epoch_end(self, epoch, logs=None):
        start_tokens = [_ for _ in self.start_tokens]
        if (epoch + 1) % self.print_every != 0:
            return
        num_tokens_generated = 0
        tokens_generated = []
        while num_tokens_generated <= self.max_tokens:
            pad_len = maxlen - len(start_tokens)
            sample_index = len(start_tokens) - 1
            if pad_len < 0:
                x = start_tokens[:maxlen]
                sample_index = maxlen - 1
            elif pad_len > 0:
                x = start_tokens + [0] * pad_len
            else:
                x = start_tokens
            x = np.array([x])
            y, _ = self.model.predict(x)
            sample_token = self.sample_from(y[0][sample_index])
            tokens_generated.append(sample_token)
            start_tokens.append(sample_token)
            num_tokens_generated = len(tokens_generated)

        txt = " ".join([
            bytes.decode(self.detokenize(_))
            for _ in self.start_tokens + tokens_generated
        ])
        print(f"generated text:\n{txt}\n")


word_to_index = {}
for index, word in enumerate(vocab):
    word_to_index[word] = index

# +
start_prompt = "this movie is"

# 需要将单词由 str 转换成 bytes，才能查词典
start_tokens = [word_to_index.get(str.encode(_), 1) for _ in start_prompt.split()]
num_tokens_generated = 40
text_gen_callback = TextGenerator(num_tokens_generated, start_tokens, vocab)
# -

# # 训练模型

model = create_model()
model.fit(text_ds, verbose=2, epochs=30, callbacks=[text_gen_callback])


