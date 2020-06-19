# -*- coding: utf-8 -*-
import copy
import json
import math
import six
import tensorflow as tf


from official.modeling import tf_utils

# # BERT模型
# `BERT`模型基于`Transformer`的编码器，由12层或更多的`EncoderLayer`组成，如下图所示，其中输入数据在模型中的维度变化如图右所示：
# <img src="../images/bert结构.png" width="80%">

# +
# huggingface 的 bert 模型
from transformers import BertTokenizer, TFBertForPreTraining

tokenizer = BertTokenizer.from_pretrained(
    '../models/bert/vocabulary.txt',  # 从保存有词汇表的本地文件载入
    do_lower_case=True)

model = TFBertForPreTraining.from_pretrained(
    "../../H/models/huggingface/bert-base-uncased/")

# 模型所有的参数，以 list 的形式
params = model.weights
# -

model.summary()





# ## 模型配置
# 创建模型时需要指定的参数：
# - `vocab_size`, 词汇表的大小，用于词嵌入矩阵
# - `hidden_size=768`, 编码层的尺寸
# - `num_hidden_layers=12`, 编码层的层数
# - `num_attention_heads=12`, 多头注意力的头数
# - `intermediate_size=3072`, 编码器中前向层的尺寸
# - `hidden_act="gelu"`, 编码器中激活函数
# - `hidden_dropout_prob=0.1`,编码器中全连接层的 dropout
# - `attention_probs_dropout_prob=0.1`, 注意力中的 dropout
# - `max_position_embeddings=512`, 最大位置数
# - `type_vocab_size=16`，文本为两个句子时的单词属于哪个句子

class BertConfig(object):
    def __init__(self, vocab_size, hidden_size=768, num_hidden_layers=12,
                 num_attention_heads=12, intermediate_size=3072,
                 hidden_act="gelu", hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1, max_position_embeddings=512,
                 type_vocab_size=16, initializer_range=0.02,
                 backward_compatible=True):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.backward_compatible = backward_compatible
    
    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size=None)
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config
    
    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with tf.io.gfile.GFile(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))
    
    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output
    
    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


# ## 整体
# 输入为：
# - 文本向量化后的 `input_ids`：每个`token`对应的id 
# - 因为文本向量填充成同样的长度对应的 `mask`：填充的`token`标记为0，文本`token`标记为1
# - 文本为两个句子时对应的 `segment_ids`：上句对应的`token`，标记为0，下句的`token`标记为1
#   
# 输出为：
# - 第一个 `token:[CLS]` 对应的输出  `pooled_output`
# - 每个 `token` 或最后一个 `token` 对应的输出 `sequence_output`

def get_bert_model(input_word_ids, input_mask, input_type_ids, config=None,
                   name=None, float_type=tf.float32):
    """Wraps the core BERT model as a keras.Model."""
    bert_model_layer = BertModel(config=config, float_type=float_type,
                                 name=name)
    pooled_output, sequence_output = bert_model_layer(input_word_ids,
                                                      input_mask,
                                                      input_type_ids)
    bert_model = tf.keras.Model(
        inputs=[input_word_ids, input_mask, input_type_ids],
        outputs=[pooled_output, sequence_output])
    return bert_model


# ## Bert框架
#
# - `embedding_lookup`，将输入单词序列转换成词嵌入向量，`[batch,seq_len]-->[batch,seq_len,hidden_size]`
# - `embedding_postprocessor`，将上一步的词嵌入矩阵，加上位置编码及句子编码
# - `encoder`，编码器，`[batch,seq_len,hidden_size]-->[batch,seq_len,hidden_size]`
# -` pooler_transform`，最终的输出层，

class BertModel(tf.keras.layers.Layer):
    """BERT model ("Bidirectional Encoder Representations from Transformers").
  
    Example usage:
  
    ```python
    # Already been converted into WordPiece token ids
    input_word_ids = tf.constant([[31, 51, 99], [15, 5, 0]])
    input_mask = tf.constant([[1, 1, 1], [1, 1, 0]])
    input_type_ids = tf.constant([[0, 0, 1], [0, 2, 0]])
  
    config = modeling.BertConfig(vocab_size=32000, hidden_size=512,
      num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)
  
    pooled_output, sequence_output = modeling.BertModel(config=config)(
      input_word_ids=input_word_ids,
      input_mask=input_mask,
      input_type_ids=input_type_ids)
    ...
    ```
    """
    
    def __init__(self, config, float_type=tf.float32, **kwargs):
        super(BertModel, self).__init__(**kwargs)
        self.config = (BertConfig.from_dict(config) if isinstance(config,
                                                                  dict) else copy.deepcopy(
            config))
        self.float_type = float_type
    
    def build(self, unused_input_shapes):
        """Implements build() for the layer."""
        self.embedding_lookup = EmbeddingLookup(
            vocab_size=self.config.vocab_size,
            embedding_size=self.config.hidden_size,
            initializer_range=self.config.initializer_range, dtype=tf.float32,
            name="word_embeddings")
        self.embedding_postprocessor = EmbeddingPostprocessor(
            use_type_embeddings=True,
            token_type_vocab_size=self.config.type_vocab_size,
            use_position_embeddings=True,
            max_position_embeddings=self.config.max_position_embeddings,
            dropout_prob=self.config.hidden_dropout_prob,
            initializer_range=self.config.initializer_range, dtype=tf.float32,
            name="embedding_postprocessor")
        self.encoder = Transformer(
            num_hidden_layers=self.config.num_hidden_layers,
            hidden_size=self.config.hidden_size,
            num_attention_heads=self.config.num_attention_heads,
            intermediate_size=self.config.intermediate_size,
            intermediate_activation=self.config.hidden_act,
            hidden_dropout_prob=self.config.hidden_dropout_prob,
            attention_probs_dropout_prob=self.config.attention_probs_dropout_prob,
            initializer_range=self.config.initializer_range,
            backward_compatible=self.config.backward_compatible,
            float_type=self.float_type, name="encoder")
        self.pooler_transform = tf.keras.layers.Dense(
            units=self.config.hidden_size, activation="tanh",
            kernel_initializer=get_initializer(self.config.initializer_range),
            name="pooler_transform")
        super(BertModel, self).build(unused_input_shapes)
    
    def __call__(self, input_word_ids, input_mask=None, input_type_ids=None,
                 **kwargs):
        inputs = tf_utils.pack_inputs(
            [input_word_ids, input_mask, input_type_ids])
        return super(BertModel, self).__call__(inputs, **kwargs)
    
    def call(self, inputs, mode="bert"):
        """Implements call() for the layer.
    
        Args:
          inputs: packed input tensors.
          mode: string, `bert` or `encoder`.
        Returns:
          Output tensor of the last layer for BERT training (mode=`bert`) which
          is a float Tensor of shape [batch_size, seq_length, hidden_size] or
          a list of output tensors for encoder usage (mode=`encoder`).
        """
        unpacked_inputs = tf_utils.unpack_inputs(inputs)
        input_word_ids = unpacked_inputs[0]
        input_mask = unpacked_inputs[1]
        input_type_ids = unpacked_inputs[2]
        
        # token ids --> token embeddings
        word_embeddings = self.embedding_lookup(input_word_ids)
        
        # 单词分属两个两个句子时的编码，及单词在句子中位置的编码
        embedding_tensor = self.embedding_postprocessor(
            word_embeddings=word_embeddings, token_type_ids=input_type_ids)
        if self.float_type == tf.float16:
            embedding_tensor = tf.cast(embedding_tensor, tf.float16)
            
        # 创建遮挡，文本转换成等长向量时，填充的token在模型中不进行计算
        attention_mask = None
        if input_mask is not None:
            attention_mask = create_attention_mask_from_input_mask(
                input_word_ids, input_mask)
        
        if mode == "encoder":
            return self.encoder(embedding_tensor, attention_mask,
                return_all_layers=True)
        
        # 编码器输出
        sequence_output = self.encoder(embedding_tensor, attention_mask)
        
        # 首个token的输出
        first_token_tensor = tf.squeeze(sequence_output[:, 0:1, :], axis=1)
        pooled_output = self.pooler_transform(first_token_tensor)
        
        return (pooled_output, sequence_output)
    
    def get_config(self):
        config = {"config": self.config.to_dict()}
        base_config = super(BertModel, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# # 输入处理
# ![](../images/bert-input.png)
# 输入由三部分组成：词嵌入、位置编码、句子编码

# ## 词嵌入

class EmbeddingLookup(tf.keras.layers.Layer):
    """Looks up words embeddings for id tensor."""
    
    def __init__(self, vocab_size, embedding_size=768, initializer_range=0.02,
                 **kwargs):
        super(EmbeddingLookup, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.initializer_range = initializer_range
    
    def build(self, unused_input_shapes):
        """Implements build() for the layer."""
        self.embeddings = self.add_weight("embeddings",
            shape=[self.vocab_size, self.embedding_size],
            initializer=get_initializer(self.initializer_range),
            dtype=self.dtype)
        super(EmbeddingLookup, self).build(unused_input_shapes)
    
    def call(self, inputs):
        """Implements call() for the layer."""
        input_shape = tf_utils.get_shape_list(inputs)
        
        # 将 betch,seq_len 的数据展平，便于计算
        flat_input = tf.reshape(inputs, [-1])
        output = tf.gather(self.embeddings, flat_input)
        
        # 再还原成 batch 数据
        output = tf.reshape(output, input_shape + [self.embedding_size])
        return output


# ## 词嵌入后处理
# - 每个单词对应的词向量 加上 该单词对应位置的位置编码
# - 如果 `use_type_embeddings=True`，就需要指定单词所属句子时的向量
# - 上述三者相加后，再对每个词向量正则化处理：
# ```
# mean_i         = sum(x_i[j] for j in range(k)) / k
# var_i          = sum((x_i[j] - mean_i) ** 2 for j in range(k)) / k
# x_i_normalized = (x_i - mean_i) / sqrt(var_i + epsilon)
# output_i       = x_i_normalized * gamma + beta
# ```
#     - gamma 和 beta 为待学习的参数

class EmbeddingPostprocessor(tf.keras.layers.Layer):
    """Performs various post-processing on a word embedding tensor."""
    
    def __init__(self, use_type_embeddings=False, token_type_vocab_size=None,
                 use_position_embeddings=True, max_position_embeddings=512,
                 dropout_prob=0.0, initializer_range=0.02, initializer=None,
                 **kwargs):
        super(EmbeddingPostprocessor, self).__init__(**kwargs)
        self.use_type_embeddings = use_type_embeddings
        self.token_type_vocab_size = token_type_vocab_size
        self.use_position_embeddings = use_position_embeddings
        self.max_position_embeddings = max_position_embeddings
        self.dropout_prob = dropout_prob
        self.initializer_range = initializer_range
        
        if not initializer:
            self.initializer = get_initializer(self.initializer_range)
        else:
            self.initializer = initializer
        
        if self.use_type_embeddings and not self.token_type_vocab_size:
            raise ValueError("If `use_type_embeddings` is True, then "
                             "`token_type_vocab_size` must be specified.")
    
    def build(self, input_shapes):
        """Implements build() for the layer."""
        (word_embeddings_shape, _) = input_shapes
        width = word_embeddings_shape.as_list()[-1]
        self.type_embeddings = None
        
        # 单词分属不同的句子时，对应一个矩阵参数，token_type_vocab_size 为句子种类数
        if self.use_type_embeddings:
            self.type_embeddings = self.add_weight("type_embeddings",
                shape=[self.token_type_vocab_size, width],
                initializer=get_initializer(self.initializer_range),
                dtype=self.dtype)
        
        # 单词再句子中的位置，对应一个矩阵参数
        self.position_embeddings = None
        if self.use_position_embeddings:
            self.position_embeddings = self.add_weight("position_embeddings",
                shape=[self.max_position_embeddings, width],
                initializer=get_initializer(self.initializer_range),
                dtype=self.dtype)
        
        # 归一化
        self.output_layer_norm = tf.keras.layers.LayerNormalization(
            name="layer_norm", axis=-1, epsilon=1e-12, dtype=tf.float32)
        self.output_dropout = tf.keras.layers.Dropout(rate=self.dropout_prob,
                                                      dtype=tf.float32)
        super(EmbeddingPostprocessor, self).build(input_shapes)
    
    def __call__(self, word_embeddings, token_type_ids=None, **kwargs):
        inputs = tf_utils.pack_inputs([word_embeddings, token_type_ids])
        return super(EmbeddingPostprocessor, self).__call__(inputs, **kwargs)
    
    def call(self, inputs):
        """Implements call() for the layer."""
        # 输入数据为 (word_embeddings,token_type_ids) 元组
        unpacked_inputs = tf_utils.unpack_inputs(inputs)
        word_embeddings = unpacked_inputs[0]
        token_type_ids = unpacked_inputs[1]
        input_shape = tf_utils.get_shape_list(word_embeddings, expected_rank=3)
        
        # 词向量: batch_size, seq_len, width
        batch_size = input_shape[0]
        seq_length = input_shape[1]
        width = input_shape[2]
        
        output = word_embeddings
        
        # 加上单词分属不同句子时的编码
        if self.use_type_embeddings:
            flat_token_type_ids = tf.reshape(token_type_ids, [-1])
            token_type_embeddings = tf.gather(self.type_embeddings,
                                              flat_token_type_ids)
            token_type_embeddings = tf.reshape(token_type_embeddings,
                                               [batch_size, seq_length, width])
            output += token_type_embeddings
        
        # 加上单词的位置编码
        if self.use_position_embeddings:
            position_embeddings = tf.expand_dims(
                tf.slice(self.position_embeddings, [0, 0], [seq_length, width]),
                axis=0)
            
            output += position_embeddings
        
        # 归一化
        output = self.output_layer_norm(output)
        
        # dropout 处理
        output = self.output_dropout(output)
        
        return output


# ## attention mask

# 创建 attention_mask
def create_attention_mask_from_input_mask(from_tensor, to_mask):
    """Create 3D attention mask from a 2D tensor mask.
  
    Args:
      from_tensor: 2D or 3D Tensor of shape [batch_size, from_seq_length, ...].
      to_mask: int32 Tensor of shape [batch_size, to_seq_length].
  
    Returns:
      float Tensor of shape [batch_size, from_seq_length, to_seq_length].
    """
    from_shape = tf_utils.get_shape_list(from_tensor, expected_rank=[2, 3])
    batch_size = from_shape[0]
    from_seq_length = from_shape[1]

    to_shape = tf_utils.get_shape_list(to_mask, expected_rank=2)
    to_seq_length = to_shape[1]

    to_mask = tf.cast(tf.reshape(to_mask, [batch_size, 1, to_seq_length]),
                      dtype=from_tensor.dtype)

    # We don't assume that `from_tensor` is a mask (although it could be). We
    # don't actually care if we attend *from* padding tokens (only *to* padding)
    # tokens so we create a tensor of all ones.
    #
    # `broadcast_ones` = [batch_size, from_seq_length, 1]
    broadcast_ones = tf.ones(shape=[batch_size, from_seq_length, 1],
                             dtype=from_tensor.dtype)

    # Here we broadcast along two dimensions to create the mask.
    mask = broadcast_ones * to_mask

    return mask



# ## 参数量

print("======输入参数及预处理=============")
for param in params[:5]:
    print('/'.join(param.name.split('/')[-2:]), param.shape, param.trainable)

# LayerNorm 中的 gamma 参数，对应词向量的维度，每个值介于 0~1 之间
params[3]

# LayerNorm 中的 beta 参数
params[4]


# - 词嵌入层，共30522个单词，每个单词768维，参数量：23440896
# - 位置编码，最长512，每个位置编码768维，参数量：393216。通常位置编码是固定的，该部分参数不用训练
# - 句子编码，共2个句子，属于每个句子的编码768维，参数量:1536
# - 正则化中 gamma 和 beta 都是768维参数

# # 编码器

# ## 多头自注意力层
# <img src="../images/self-attn.png" width="75%">

class Attention(tf.keras.layers.Layer):
    """Performs multi-headed attention from `from_tensor` to `to_tensor`.
  
    This is an implementation of multi-headed attention based on "Attention
    is all you Need". If `from_tensor` and `to_tensor` are the same, then
    this is self-attention. Each timestep in `from_tensor` attends to the
    corresponding sequence in `to_tensor`, and returns a fixed-with vector.
  
    This function first projects `from_tensor` into a "query" tensor and
    `to_tensor` into "key" and "value" tensors. These are (effectively) a list
    of tensors of length `num_attention_heads`, where each tensor is of shape
    [batch_size, seq_length, size_per_head].
  
    Then, the query and key tensors are dot-producted and scaled. These are
    softmaxed to obtain attention probabilities. The value tensors are then
    interpolated by these probabilities, then concatenated back to a single
    tensor and returned.
  
    In practice, the multi-headed attention are done with tf.einsum as follows:
      Input_tensor: [BFD]
      Wq, Wk, Wv: [DNH]
      Q:[BFNH] = einsum('BFD,DNH->BFNH', Input_tensor, Wq)
      K:[BTNH] = einsum('BTD,DNH->BTNH', Input_tensor, Wk)
      V:[BTNH] = einsum('BTD,DNH->BTNH', Input_tensor, Wv)
      attention_scores:[BNFT] = einsum('BTNH,BFNH->BNFT', K, Q) / sqrt(H)
      attention_probs:[BNFT] = softmax(attention_scores)
      context_layer:[BFNH] = einsum('BNFT,BTNH->BFNH', attention_probs, V)
      Wout:[DNH]
      Output:[BFD] = einsum('BFNH,DNH>BFD', context_layer, Wout)
    """
    
    def __init__(self, num_attention_heads=12, size_per_head=64,
                 attention_probs_dropout_prob=0.0, initializer_range=0.02,
                 backward_compatible=False, **kwargs):
        super(Attention, self).__init__(**kwargs)
        
        # 注意力的头数
        self.num_attention_heads = num_attention_heads
        
        # 每个头的维度
        self.size_per_head = size_per_head
        
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.backward_compatible = backward_compatible
    
    def build(self, unused_input_shapes):
        """Implements build() for the layer."""
        # 将输入转换成 query，key，value
        self.query_dense = self._projection_dense_layer("query")
        self.key_dense = self._projection_dense_layer("key")
        self.value_dense = self._projection_dense_layer("value")
        self.attention_probs_dropout = tf.keras.layers.Dropout(
            rate=self.attention_probs_dropout_prob)
        super(Attention, self).build(unused_input_shapes)
    
    def reshape_to_matrix(self, input_tensor):
        """Reshape N > 2 rank tensor to rank 2 tensor for performance."""
        ndims = input_tensor.shape.ndims
        if ndims < 2:
            raise ValueError("Input tensor must have at least rank 2."
                             "Shape = %s" % (input_tensor.shape))
        if ndims == 2:
            return input_tensor
        
        width = input_tensor.shape[-1]
        output_tensor = tf.reshape(input_tensor, [-1, width])
        return output_tensor
    
    def __call__(self, from_tensor, to_tensor, attention_mask=None, **kwargs):
        inputs = tf_utils.pack_inputs([from_tensor, to_tensor, attention_mask])
        return super(Attention, self).__call__(inputs, **kwargs)
    
    def call(self, inputs):
        """Implements call() for the layer."""
        (from_tensor, to_tensor, attention_mask) = tf_utils.unpack_inputs(
            inputs)
        
        # Scalar dimensions referenced here:
        #   B = batch size (number of sequences)
        #   F = `from_tensor` sequence length
        #   T = `to_tensor` sequence length
        #   N = `num_attention_heads`
        #   H = `size_per_head`
        # `query_tensor` = [B, F, N ,H]
        query_tensor = self.query_dense(from_tensor)
        
        # `key_tensor` = [B, T, N, H]
        key_tensor = self.key_dense(to_tensor)
        
        # `value_tensor` = [B, T, N, H]
        value_tensor = self.value_dense(to_tensor)
        
        # Take the dot product between "query" and "key" to get the raw
        # attention scores.
        attention_scores = tf.einsum("BTNH,BFNH->BNFT", key_tensor,
                                     query_tensor)
        attention_scores = tf.multiply(attention_scores, 1.0 / math.sqrt(
            float(self.size_per_head)))
        
        if attention_mask is not None:
            # `attention_mask` = [B, 1, F, T]
            attention_mask = tf.expand_dims(attention_mask, axis=[1])
            
            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            adder = (1.0 - tf.cast(attention_mask,
                                   attention_scores.dtype)) * -10000.0
            
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_scores += adder
        
        # Normalize the attention scores to probabilities.
        # `attention_probs` = [B, N, F, T]
        attention_probs = tf.nn.softmax(attention_scores)
        
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.attention_probs_dropout(attention_probs)
        
        # `context_layer` = [B, F, N, H]
        context_tensor = tf.einsum("BNFT,BTNH->BFNH", attention_probs,
                                   value_tensor)
        
        return context_tensor
    
    def _projection_dense_layer(self, name):
        """A helper to define a projection layer."""
        return Dense3D(num_attention_heads=self.num_attention_heads,
            size_per_head=self.size_per_head,
            kernel_initializer=get_initializer(self.initializer_range),
            output_projection=False,
            backward_compatible=self.backward_compatible, name=name)


class Dense3D(tf.keras.layers.Layer):
    """A Dense Layer using 3D kernel with tf.einsum implementation.
  
    Attributes:
      num_attention_heads: An integer, number of attention heads for each
        multihead attention layer.
      size_per_head: An integer, hidden size per attention head.
      hidden_size: An integer, dimension of the hidden layer.
      kernel_initializer: An initializer for the kernel weight.
      bias_initializer: An initializer for the bias.
      activation: An activation function to use. If nothing is specified, no
        activation is applied.
      use_bias: A bool, whether the layer uses a bias.
      output_projection: A bool, whether the Dense3D layer is used for output
        linear projection.
      backward_compatible: A bool, whether the variables shape are compatible
        with checkpoints converted from TF 1.x.
    """
    
    def __init__(self, num_attention_heads=12, size_per_head=72,
                 kernel_initializer=None, bias_initializer="zeros",
                 activation=None, use_bias=True, output_projection=False,
                 backward_compatible=False, **kwargs):
        """Inits Dense3D."""
        super(Dense3D, self).__init__(**kwargs)
        self.num_attention_heads = num_attention_heads
        self.size_per_head = size_per_head
        self.hidden_size = num_attention_heads * size_per_head
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.activation = activation
        self.use_bias = use_bias
        self.output_projection = output_projection
        self.backward_compatible = backward_compatible
    
    @property
    def compatible_kernel_shape(self):
        if self.output_projection:
            return [self.hidden_size, self.hidden_size]
        return [self.last_dim, self.hidden_size]
    
    @property
    def compatible_bias_shape(self):
        return [self.hidden_size]
    
    @property
    def kernel_shape(self):
        if self.output_projection:
            return [self.num_attention_heads, self.size_per_head,
                    self.hidden_size]
        return [self.last_dim, self.num_attention_heads, self.size_per_head]
    
    @property
    def bias_shape(self):
        if self.output_projection:
            return [self.hidden_size]
        return [self.num_attention_heads, self.size_per_head]
    
    def build(self, input_shape):
        """Implements build() for the layer."""
        dtype = tf.as_dtype(self.dtype or tf.keras.backend.floatx())
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError("Unable to build `Dense3D` layer with non-floating "
                            "point (and non-complex) dtype %s" % (dtype,))
        input_shape = tf.TensorShape(input_shape)
        if tf.compat.dimension_value(input_shape[-1]) is None:
            raise ValueError("The last dimension of the inputs to `Dense3D` "
                             "should be defined. Found `None`.")
        self.last_dim = tf.compat.dimension_value(input_shape[-1])
        self.input_spec = tf.keras.layers.InputSpec(min_ndim=3,
            axes={-1: self.last_dim})
        # Determines variable shapes.
        if self.backward_compatible:
            kernel_shape = self.compatible_kernel_shape
            bias_shape = self.compatible_bias_shape
        else:
            kernel_shape = self.kernel_shape
            bias_shape = self.bias_shape
        
        self.kernel = self.add_weight("kernel", shape=kernel_shape,
            initializer=self.kernel_initializer, dtype=self.dtype,
            trainable=True)
        if self.use_bias:
            self.bias = self.add_weight("bias", shape=bias_shape,
                initializer=self.bias_initializer, dtype=self.dtype,
                trainable=True)
        else:
            self.bias = None
        super(Dense3D, self).build(input_shape)
    
    def call(self, inputs):
        """Implements ``call()`` for Dense3D.
    
        Args:
          inputs: A float tensor of shape [batch_size, sequence_length, hidden_size]
            when output_projection is False, otherwise a float tensor of shape
            [batch_size, sequence_length, num_heads, dim_per_head].
    
        Returns:
          The projected tensor with shape [batch_size, sequence_length, num_heads,
            dim_per_head] when output_projection is False, otherwise [batch_size,
            sequence_length, hidden_size].
        """
        if self.backward_compatible:
            kernel = tf.keras.backend.reshape(self.kernel, self.kernel_shape)
            bias = (tf.keras.backend.reshape(self.bias,
                                             self.bias_shape) if self.use_bias else None)
        else:
            kernel = self.kernel
            bias = self.bias
        
        if self.output_projection:
            ret = tf.einsum("abcd,cde->abe", inputs, kernel)
        else:
            ret = tf.einsum("abc,cde->abde", inputs, kernel)
        if self.use_bias:
            ret += bias
        if self.activation is not None:
            return self.activation(ret)
        return ret


class Dense2DProjection(tf.keras.layers.Layer):
    """A 2D projection layer with tf.einsum implementation."""
    
    def __init__(self, output_size, kernel_initializer=None,
                 bias_initializer="zeros", activation=None,
                 fp32_activation=False, **kwargs):
        super(Dense2DProjection, self).__init__(**kwargs)
        self.output_size = output_size
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.activation = activation
        self.fp32_activation = fp32_activation
    
    def build(self, input_shape):
        """Implements build() for the layer."""
        dtype = tf.as_dtype(self.dtype or tf.keras.backend.floatx())
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError("Unable to build `Dense2DProjection` layer with "
                            "non-floating point (and non-complex) "
                            "dtype %s" % (dtype,))
        input_shape = tf.TensorShape(input_shape)
        if tf.compat.dimension_value(input_shape[-1]) is None:
            raise ValueError("The last dimension of the inputs to "
                             "`Dense2DProjection` should be defined. "
                             "Found `None`.")
        last_dim = tf.compat.dimension_value(input_shape[-1])
        self.input_spec = tf.keras.layers.InputSpec(min_ndim=3,
                                                    axes={-1: last_dim})
        self.kernel = self.add_weight("kernel",
            shape=[last_dim, self.output_size],
            initializer=self.kernel_initializer, dtype=self.dtype,
            trainable=True)
        self.bias = self.add_weight("bias", shape=[self.output_size],
            initializer=self.bias_initializer, dtype=self.dtype, trainable=True)
        super(Dense2DProjection, self).build(input_shape)
    
    def call(self, inputs):
        """Implements call() for Dense2DProjection.
    
        Args:
          inputs: float Tensor of shape [batch, from_seq_length,
            num_attention_heads, size_per_head].
    
        Returns:
          A 3D Tensor.
        """
        ret = tf.einsum("abc,cd->abd", inputs, self.kernel)
        ret += self.bias
        if self.activation is not None:
            if self.dtype == tf.float16 and self.fp32_activation:
                ret = tf.cast(ret, tf.float32)
            return self.activation(ret)
        return ret


# ### 注意力层参数量

print("======注意力层参数=============")
for param in params[5:15]:
    print('/'.join(param.name.split('/')[2:]), param.shape, param.trainable)

# 编码器，第一层，注意力的参数：
# - 将输入转换成 query、key、value 时，**三组** 权重+偏置 参数，分别为 (768,768) 和（768,)
# - 然后进行点积注意力运算后的输出，再经过密集层处理：权重+偏置 参数，分别为 (768,768) 和（768,)
# - 将上述 **输出和注意力的输入相加** 后，再次正则化， gamma 和 beta 都是768维参数

['/'.join(param.name.split('/')[2:]) for param in model.weights]









# ## 编码层

# 其层次结构如下：
# ```
# 1. attention_layer       :[batch,seq,hidden_size]-->[batch,seq,hidden_size]
# 2. attention_output_dense  
# 3. attention_dropout
# 4. layer_norm
# 5. intermediate_dense    :[batch,seq,hidden_size]-->[batch,seq,intermediate_size]
# 6. output_dense          :[batch,seq,intermediate_size]-->[batch,seq,hidden_size]
# 7. output_dropout
# 8. layer_norm            
# ```

class TransformerBlock(tf.keras.layers.Layer):
    """Single transformer layer.
  
    It has two sub-layers. The first is a multi-head self-attention mechanism, and
    the second is a positionwise fully connected feed-forward network.
    """
    
    def __init__(self, hidden_size=768, num_attention_heads=12,
                 intermediate_size=3072, intermediate_activation="gelu",
                 hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0,
                 initializer_range=0.02, backward_compatible=False,
                 float_type=tf.float32, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.intermediate_activation = tf_utils.get_activation(
            intermediate_activation)
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.backward_compatible = backward_compatible
        self.float_type = float_type
        
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (self.hidden_size, self.num_attention_heads))
        self.attention_head_size = int(
            self.hidden_size / self.num_attention_heads)
    
    def build(self, unused_input_shapes):
        """Implements build() for the layer."""
        self.attention_layer = Attention(
            num_attention_heads=self.num_attention_heads,
            size_per_head=self.attention_head_size,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            initializer_range=self.initializer_range,
            backward_compatible=self.backward_compatible, name="self_attention")
        self.attention_output_dense = Dense3D(
            num_attention_heads=self.num_attention_heads,
            size_per_head=int(self.hidden_size / self.num_attention_heads),
            kernel_initializer=get_initializer(self.initializer_range),
            output_projection=True,
            backward_compatible=self.backward_compatible,
            name="self_attention_output")
        self.attention_dropout = tf.keras.layers.Dropout(
            rate=self.hidden_dropout_prob)
        self.attention_layer_norm = (
            tf.keras.layers.LayerNormalization(name="self_attention_layer_norm",
                axis=-1, epsilon=1e-12,
                # We do layer norm in float32 for numeric stability.
                dtype=tf.float32))
        self.intermediate_dense = Dense2DProjection(
            output_size=self.intermediate_size,
            kernel_initializer=get_initializer(self.initializer_range),
            activation=self.intermediate_activation,
            # Uses float32 so that gelu activation is done in float32.
            fp32_activation=True, name="intermediate")
        
        self.output_dense = Dense2DProjection(output_size=self.hidden_size,
            kernel_initializer=get_initializer(self.initializer_range),
            name="output")
        self.output_dropout = tf.keras.layers.Dropout(
            rate=self.hidden_dropout_prob)
        self.output_layer_norm = tf.keras.layers.LayerNormalization(
            name="output_layer_norm", axis=-1, epsilon=1e-12, dtype=tf.float32)
        super(TransformerBlock, self).build(unused_input_shapes)
    
    def common_layers(self):
        """Explicitly gets all layer objects inside a Transformer encoder block."""
        return [self.attention_layer, self.attention_output_dense,
            self.attention_dropout, self.attention_layer_norm,
            self.intermediate_dense, self.output_dense, self.output_dropout,
            self.output_layer_norm]
    
    def __call__(self, input_tensor, attention_mask=None, **kwargs):
        inputs = tf_utils.pack_inputs([input_tensor, attention_mask])
        return super(TransformerBlock, self).__call__(inputs, **kwargs)
    
    def call(self, inputs):
        """Implements call() for the layer."""
        (input_tensor, attention_mask) = tf_utils.unpack_inputs(inputs)
        attention_output = self.attention_layer(from_tensor=input_tensor,
            to_tensor=input_tensor, attention_mask=attention_mask)
        attention_output = self.attention_output_dense(attention_output)
        attention_output = self.attention_dropout(attention_output)
        # Use float32 in keras layer norm and the gelu activation in the
        # intermediate dense layer for numeric stability
        attention_output = self.attention_layer_norm(
            input_tensor + attention_output)
        if self.float_type == tf.float16:
            attention_output = tf.cast(attention_output, tf.float16)
        intermediate_output = self.intermediate_dense(attention_output)
        if self.float_type == tf.float16:
            intermediate_output = tf.cast(intermediate_output, tf.float16)
        layer_output = self.output_dense(intermediate_output)
        layer_output = self.output_dropout(layer_output)
        # Use float32 in keras layer norm for numeric stability
        layer_output = self.output_layer_norm(layer_output + attention_output)
        if self.float_type == tf.float16:
            layer_output = tf.cast(layer_output, tf.float16)
        return layer_output


# ### 编码层参数量

print("======编码层参数=============")
for param in params[5:21]:
    print('/'.join(param.name.split('/')[2:]), param.shape, param.trainable)


# 上述注意力的输出后
# - 经过前向密集层：权重 (768, 3072) ，偏置 (3072,) 
# - 密集层：权重 (3072,768) ，偏置 (768,) 
# - 再次与注意力的输出相加后，正则化： gamma 和 beta 都是768维参数

# ## 编码器
# 由上一步的多层编码层组成

class Transformer(tf.keras.layers.Layer):
    """Multi-headed, multi-layer Transformer from "Attention is All You Need".
  
    This is almost an exact implementation of the original Transformer encoder.
  
    See the original paper:
    https://arxiv.org/abs/1706.03762
  
    Also see:
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py
    """
    
    def __init__(self, num_hidden_layers=12, hidden_size=768,
                 num_attention_heads=12, intermediate_size=3072,
                 intermediate_activation="gelu", hidden_dropout_prob=0.0,
                 attention_probs_dropout_prob=0.0, initializer_range=0.02,
                 backward_compatible=False, float_type=tf.float32, **kwargs):
        super(Transformer, self).__init__(**kwargs)
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.intermediate_activation = tf_utils.get_activation(
            intermediate_activation)
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.backward_compatible = backward_compatible
        self.float_type = float_type
    
    def build(self, unused_input_shapes):
        """Implements build() for the layer."""
        self.layers = []
        for i in range(self.num_hidden_layers):
            self.layers.append(TransformerBlock(hidden_size=self.hidden_size,
                num_attention_heads=self.num_attention_heads,
                intermediate_size=self.intermediate_size,
                intermediate_activation=self.intermediate_activation,
                hidden_dropout_prob=self.hidden_dropout_prob,
                attention_probs_dropout_prob=self.attention_probs_dropout_prob,
                initializer_range=self.initializer_range,
                backward_compatible=self.backward_compatible,
                float_type=self.float_type, name=("layer_%d" % i)))
        super(Transformer, self).build(unused_input_shapes)
    
    def __call__(self, input_tensor, attention_mask=None, **kwargs):
        inputs = tf_utils.pack_inputs([input_tensor, attention_mask])
        return super(Transformer, self).__call__(inputs=inputs, **kwargs)
    
    def call(self, inputs, return_all_layers=False):
        """Implements call() for the layer.
    
        Args:
          inputs: packed inputs.
          return_all_layers: bool, whether to return outputs of all layers inside
            encoders.
        Returns:
          Output tensor of the last layer or a list of output tensors.
        """
        unpacked_inputs = tf_utils.unpack_inputs(inputs)
        input_tensor = unpacked_inputs[0]
        attention_mask = unpacked_inputs[1]
        output_tensor = input_tensor
        
        all_layer_outputs = []
        for layer in self.layers:
            output_tensor = layer(output_tensor, attention_mask)
            all_layer_outputs.append(output_tensor)
        
        if return_all_layers:
            return all_layer_outputs
        
        return all_layer_outputs[-1]


# ### 编码器的参数

print("======编码期参数=============")
for param in params[5:197]:
    print('/'.join(param.name.split('/')[2:]), param.shape, param.trainable)

# **相同结构的编码层重复12遍**

for param in params[197:]:
    print('/'.join(param.name.split('/')[2:]), param.shape, param.trainable)





# # 辅助函数

# 初始化函数
def get_initializer(initializer_range=0.02):
    """Creates a `tf.initializers.truncated_normal` with the given range.
  
    Args:
      initializer_range: float, initializer range for stddev.
  
    Returns:
      TruncatedNormal initializer with stddev = `initializer_range`.
    """
    return tf.keras.initializers.TruncatedNormal(stddev=initializer_range)




