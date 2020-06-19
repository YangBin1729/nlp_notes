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
"""Keras-based einsum layer."""
# pylint: disable=g-classes-have-attributes
from __future__ import absolute_import
from __future__ import division
# from __future__ import google_type_annotations
from __future__ import print_function

import tensorflow as tf

_CHR_IDX = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m"]

"""
output_shape (d,e), num_summed=1, 输入(a,b,c)，对应的权重(c,d,e) --> abde
output_shape (d,e), num_summed=2, 输入(a,b,c)，对应的权重(b,c,d,e) --> ade
"""


@tf.keras.utils.register_keras_serializable(package="Text")
class DenseEinsum(tf.keras.layers.Layer):
    """A densely connected layer that uses tf.einsum as the backing computation.
  
    This layer can perform einsum calculations of arbitrary dimensionality.
  
    Arguments:
      output_shape: Positive integer or tuple, dimensionality of the output space.
      num_summed_dimensions: The number of dimensions to sum over. Standard 2D
        matmul should use 1, 3D matmul should use 2, and so forth.
      activation: Activation function to use. If you don't specify anything, no
        activation is applied
        (ie. "linear" activation: `a(x) = x`).
      use_bias: Boolean, whether the layer uses a bias vector.
      kernel_initializer: Initializer for the `kernel` weights matrix.
      bias_initializer: Initializer for the bias vector.
      kernel_regularizer: Regularizer function applied to the `kernel` weights
        matrix.
      bias_regularizer: Regularizer function applied to the bias vector.
      activity_regularizer: Regularizer function applied to the output of the
        layer (its "activation")..
      kernel_constraint: Constraint function applied to the `kernel` weights
        matrix.
      bias_constraint: Constraint function applied to the bias vector.
    Input shape:
      N-D tensor with shape: `(batch_size, ..., input_dim)`. The most common
        situation would be a 2D input with shape `(batch_size, input_dim)`.
    Output shape:
      N-D tensor with shape: `(batch_size, ..., units)`. For instance, for a 2D
        input with shape `(batch_size, input_dim)`, the output would have shape
        `(batch_size, units)`.
    """
    
    def __init__(self,
                 output_shape,
                 num_summed_dimensions=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer="glorot_uniform",
                 bias_initializer="zeros",
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        
        super(DenseEinsum, self).__init__(**kwargs)
        
        # 输出形状
        self._output_shape = output_shape if isinstance(output_shape, (list, tuple)) \
            else (output_shape,)
        
        # 输出前的激活函数
        self._activation = tf.keras.activations.get(activation)
        
        # 是否使用偏置
        self._use_bias = use_bias
        
        # 初始化
        self._kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self._bias_initializer = tf.keras.initializers.get(bias_initializer)
        
        # 正则化
        self._kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self._bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        
        # 限制权重
        self._kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self._bias_constraint = tf.keras.constraints.get(bias_constraint)
        
        # 需要求和的维度数量，'aijk,akl->akl'，则为 2
        self._num_summed_dimensions = num_summed_dimensions
        self._einsum_string = None
    
    def _build_einsum_string(self, free_input_dims, bound_dims, output_dims):
        input_str = ""
        kernel_str = ""
        output_str = ""
        letter_offset = 0
        
        # 输入和输出都必须具备的维度
        for i in range(free_input_dims):
            char = _CHR_IDX[i + letter_offset]
            input_str += char            # a,b
            output_str += char           # a,b
            
        # 权重的维度
        letter_offset += free_input_dims
        for i in range(bound_dims):
            char = _CHR_IDX[i + letter_offset]
            input_str += char           # a,b,c,d
            kernel_str += char          # c,d
        
        # 输出的维度
        letter_offset += bound_dims
        for i in range(output_dims):
            char = _CHR_IDX[i + letter_offset]
            kernel_str += char         # c,d,e
            output_str += char         # a,b,e
        
        return input_str + "," + kernel_str + "->" + output_str
    
    def build(self, input_shape):
        # 层实例化之后，初次调用 call 方法时，自动执行该方法
        # input_shape 由初次调用call方法时的输入自动计算得到
        input_shape = tf.TensorShape(input_shape)                   # a,b,c
        input_rank = input_shape.rank                               # 2
        free_input_dims = input_rank - self._num_summed_dimensions  # 3 - 1
        output_dims = len(self._output_shape)                       # d,e
        
        # 'abc,bcde->abde'
        self._einsum_string = self._build_einsum_string(free_input_dims,
                                                        self._num_summed_dimensions,
                                                        output_dims)
        
        # This is only saved for testing purposes.
        # c,d,e
        self._kernel_shape = (
            input_shape[free_input_dims:].concatenate(self._output_shape))
        
        # 定义权重参数
        self._kernel = self.add_weight(
            "kernel",
            shape=self._kernel_shape,
            initializer=self._kernel_initializer,
            regularizer=self._kernel_regularizer,
            constraint=self._kernel_constraint,
            dtype=self.dtype,
            trainable=True)
        
        # 定义偏置参数
        if self._use_bias:
            self._bias = self.add_weight(
                "bias",
                shape=self._output_shape,
                initializer=self._bias_initializer,
                regularizer=self._bias_regularizer,
                constraint=self._bias_constraint,
                dtype=self.dtype,
                trainable=True)
        else:
            self._bias = None
        super(DenseEinsum, self).build(input_shape)
    
    def get_config(self):
        config = {
            "output_shape":
                self._output_shape,
            "num_summed_dimensions":
                self._num_summed_dimensions,
            "activation":
                tf.keras.activations.serialize(self._activation),
            "use_bias":
                self._use_bias,
            "kernel_initializer":
                tf.keras.initializers.serialize(self._kernel_initializer),
            "bias_initializer":
                tf.keras.initializers.serialize(self._bias_initializer),
            "kernel_regularizer":
                tf.keras.regularizers.serialize(self._kernel_regularizer),
            "bias_regularizer":
                tf.keras.regularizers.serialize(self._bias_regularizer),
            "activity_regularizer":
                tf.keras.regularizers.serialize(self._activity_regularizer),
            "kernel_constraint":
                tf.keras.constraints.serialize(self._kernel_constraint),
            "bias_constraint":
                tf.keras.constraints.serialize(self._bias_constraint)
        }
        base_config = super(DenseEinsum, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def call(self, inputs):
        # einsum 计算
        ret = tf.einsum(self._einsum_string, inputs, self._kernel)
        # 使用偏置
        if self._use_bias:
            ret += self._bias
        # 使用激活函数
        if self._activation is not None:
            ret = self._activation(ret)
        return ret