# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Keras-based positional embedding layer."""
# pylint: disable=g-classes-have-attributes
from __future__ import absolute_import
from __future__ import division
# from __future__ import google_type_annotations
from __future__ import print_function

import math

import tensorflow as tf

from official.modeling import tf_utils


# 通过输入编码(batch,seq_len,embed_size)，获得绝对位置编码（batch,seq_len,position_embed_size)
@tf.keras.utils.register_keras_serializable(package="Text")
class PositionEmbedding(tf.keras.layers.Layer):
    """Creates a positional embedding.
  
    This layer creates a positional embedding as described in "BERT: Pre-training
    of Deep Bidirectional Transformers for Language Understanding"
    (https://arxiv.org/abs/1810.04805).
  
    This layer can be set up to either create a statically shaped slice or a
    dynamically shaped slice. If `use_dynamic_slicing` is True, the input tensor
    can have a dynamic 1st dimension, while if `use_dynamic_slicing` is False the
    input size must be fixed.
  
    Arguments:
      use_dynamic_slicing: Whether to use the dynamic slicing path.
      max_sequence_length: The maximum size of the dynamic sequence. Only
        applicable if `use_dynamic_slicing` is True.
      initializer: The initializer to use for the embedding weights. Defaults to
        "glorot_uniform".
    """
    
    def __init__(self,
                 initializer="glorot_uniform",
                 use_dynamic_slicing=False,
                 max_sequence_length=None,
                 **kwargs):
        # We need to have a default dtype of float32, since the inputs (which Keras
        # usually uses to infer the dtype) will always be int32.
        if "dtype" not in kwargs:
            kwargs["dtype"] = "float32"
        
        super(PositionEmbedding, self).__init__(**kwargs)
        
        # 动态切片时，必须指定嵌入矩阵最大长度，如 512
        if use_dynamic_slicing and max_sequence_length is None:
            raise ValueError(
                "If `use_dynamic_slicing` is True, `max_sequence_length` must be set."
            )
        self._max_sequence_length = max_sequence_length
        self._initializer = tf.keras.initializers.get(initializer)
        self._use_dynamic_slicing = use_dynamic_slicing
    
    def get_config(self):
        config = {
            "max_sequence_length": self._max_sequence_length,
            "initializer": tf.keras.initializers.serialize(self._initializer),
            "use_dynamic_slicing": self._use_dynamic_slicing,
        }
        base_config = super(PositionEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    # 初始化层后第一次调用时，才执行该方法；再重复使用层时，不会执行
    def build(self, input_shape):
        """Implements build() for the layer."""
        dimension_list = input_shape.as_list()
        
        if len(dimension_list) != 3:
            raise ValueError("PositionEmbedding expects a 3-dimensional input tensor "
                             "of shape [batch, sequence, width]")
        seq_length = dimension_list[1]
        width = dimension_list[2]
        
        # If we are not using dynamic slicing, we must assume that the sequence
        # length is fixed and max_sequence_length should not be specified.
        if not self._use_dynamic_slicing:
            if seq_length is None:
                raise ValueError(
                    "PositionEmbedding must have `use_dynamic_slicing` set "
                    "to True (and max_sequence_length set) when the "
                    "sequence (1st) dimension of the input is None.")
            if self._max_sequence_length is not None:
                raise ValueError(
                    "When `use_dynamic_slicing` is False, max_sequence_length should "
                    "not be specified and we ought to use seq_length to get the "
                    "variable shape.")
        
        if self._max_sequence_length is not None:
            weight_sequence_length = self._max_sequence_length
        else:
            weight_sequence_length = seq_length
        
        # 静态切片时，第一次调用层时，根据输入的长度，创建嵌入矩阵
        # 再次调用时，如果输入的长度大于嵌入矩阵的长度，也只会返回嵌入矩阵
        # 嵌入矩阵的维度，是根据输入的维度，自动确定，不需要指定
        self._position_embeddings = self.add_weight(
            "embeddings",
            shape=[weight_sequence_length, width],
            initializer=self._initializer)
        
        super(PositionEmbedding, self).build(input_shape)
    
    def call(self, inputs):
        """Implements call() for the layer."""
        # 动态切片
        # 嵌入矩阵是固定的 (max_seq_len,embed_width)，从中截取目标长度的位置向量 （seq_len,width)
        input_shape = tf_utils.get_shape_list(inputs, expected_rank=3)
        if self._use_dynamic_slicing:
            position_embeddings = self._position_embeddings[:input_shape[1], :]
        
        # 静态切片时，第一次调用层时，根据输入（batch,seq_len,width）的长度，创建嵌入矩阵 （seq_len,width）
        # 再次调用时，不管输入的长度多少，也只会返回嵌入矩阵本身 （seq_len,width）
        else:
            position_embeddings = self._position_embeddings
        
        # (seq,width) --> (batch,seq,width)
        return tf.broadcast_to(position_embeddings, input_shape)


# 相对位置编码
@tf.keras.utils.register_keras_serializable(package="Text")
class RelativePositionEmbedding(tf.keras.layers.Layer):
    """Creates a positional embedding.
  
    This layer calculates the position encoding as a mix of sine and cosine
    functions with geometrically increasing wavelengths. Defined and formulized in
     "Attention is All You Need", section 3.5.
    (https://arxiv.org/abs/1706.03762).
  
    Arguments:
      hidden_size: Size of the hidden layer.
      min_timescale: Minimum scale that will be applied at each position
      max_timescale: Maximum scale that will be applied at each position.
    """
    
    def __init__(self,
                 hidden_size,
                 min_timescale=1.0,
                 max_timescale=1.0e4,
                 **kwargs):
        # We need to have a default dtype of float32, since the inputs (which Keras
        # usually uses to infer the dtype) will always be int32.
        # We compute the positional encoding in float32 even if the model uses
        # float16, as many of the ops used, like log and exp, are numerically
        # unstable in float16.
        if "dtype" not in kwargs:
            kwargs["dtype"] = "float32"
        
        super(RelativePositionEmbedding, self).__init__(**kwargs)
        self._hidden_size = hidden_size
        self._min_timescale = min_timescale
        self._max_timescale = max_timescale
    
    def get_config(self):
        config = {
            "hidden_size": self._hidden_size,
            "min_timescale": self._min_timescale,
            "max_timescale": self._max_timescale,
            "length": self._length,
        }
        base_config = super(RelativePositionEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def call(self, inputs, length=None):
        """Implements call() for the layer.
    
        Args:
          inputs: An tensor whose second dimension will be used as `length`. If
            `None`, the other `length` argument must be specified.
          length: An optional integer specifying the number of positions. If both
            `inputs` and `length` are spcified, `length` must be equal to the
            second dimension of `inputs`.
    
        Returns:
          A tensor in shape of [length, hidden_size].
        """
        if inputs is None and length is None:
            raise ValueError(
                "If inputs is None, `length` must be set in "
                "RelativePositionEmbedding().")
        if inputs is not None:
            input_shape = tf_utils.get_shape_list(inputs)
            if length is not None and length != input_shape[1]:
                raise ValueError(
                    "If inputs is not None, `length` must equal to input_shape[1]."
                )
            length = input_shape[1]
        
        # range(10)
        position = tf.cast(tf.range(length), tf.float32)
        
        # e.g. : 8 // 2
        num_timescales = self._hidden_size // 2
        
        # 1.0, 1.0e4
        min_timescale, max_timescale = self._min_timescale, self._max_timescale
        
        # log(1.e4 / 1.) / (4 - 1) = 3.07
        log_timescale_increment = (
                math.log(float(max_timescale) / float(min_timescale)) /
                (tf.cast(num_timescales, tf.float32) - 1))
        
        # 1.0 * exp( [0.0, 1.0, 2.0, 3.0 ] * -3.07 )
        inv_timescales = min_timescale * tf.exp(
            tf.cast(tf.range(num_timescales), tf.float32) * -log_timescale_increment
        )
        
        # (length,1) * (1,num_timescale)
        scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales,
                                                                   0)
        # 分别 sin 和 cos 操作，然后拼接成 hidden_size 长的向量
        position_embeddings = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)],
                                        axis=1)
        
        # r = log( max_ - min_) / (hidden_size / 2)
        # a = [ [0], [1], [2]...[len] ] * [ e^ ( r * [0, 1, 2, ... hidden_size/2 ] ) ]
        # o = concat( sin(a), cos(a) )  -->  len, hidden_size
        return position_embeddings