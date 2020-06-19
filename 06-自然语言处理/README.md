# 自然语言处理学习笔记

机器学习及深度学习原理和示例，基于 Tensorflow 和 PyTorch 框架，Transformer、BERT、ALBERT等最新预训练模型及源代码详解，及基于预训练模型进行各种自然语言处理任务。以及模型部署



## [01-传统模型](./01-传统模型) 

两种传统的模型：

- [01-基于规则与基于概率的模型](./01-传统模型/01-基于规则与基于概率的模型.ipynb)
  - 基于规则或模板生成对话系统
  - 基于概率的语言模型
      - 利用语料数据，实现了简略的 2-gram  模型，并利用该模型判断句子的合理性

- [02-基于搜索的决策系统.ipynb](./01-传统模型/02-基于搜索的决策系统.ipynb )
  - 根据中国城市的位置信息，实现简单的路径规划系统
  - 根据武汉地铁的各站点的位置信息，实现简单的路径规划系统

      - 图的广度优先搜索及深度优先搜索
  - 搜索问题的抽象模式
      - Travelling Sales man Problem
          - 启发式    
          
          - A* 搜索
          
          - 动态规划
          

## [02-机器学习](./02-机器学习)

- 机器学习算法，及其应用

  

## [03-神经网络Python实现](./03-神经网络Python实现)

- python 实现基本的神经网络：激活函数，损失函数，前向传播，反向传播
- python 实现各种梯度下降算法，初始化，Batch Normalization，正则化
- python 实行 CNN 



## [04-深度学习框架](./04-深度学习框架)

- [TensorFlow](./04-深度学习框架/TensorFlow) ：           
-  [01-TensorFlow张量与自动微分](./04-深度学习框架/TensorFlow/01-TensorFlow张量与自动微分.ipynb)
    - Tensor Flow 基本概念，张量，张量运算，自动微分，及 tf.function 和 AutoGraph 使用原理
  
- [02-TensorFlow数据管道及特征列](./04-深度学习框架/TensorFlow/02-TensorFlow数据管道及特征列.ipynb)
    - TensorFlow 的数据管道，利用 tf.data.Dataset 预处理数据，提升性能
    - TensorFlow 内置的特征函数，用于特征工程
  
- [03-TensorFlow高阶API](./04-深度学习框架/TensorFlow/03-TensorFlow高阶API.ipynb)
    - 三种创建模型方法：Sequential、函数式、tf.keras.Model子类化
    - 三种模型训练方法：模型的 fit 方法，train_on_batch 方法，利用 tf.GradientTape自定义训练循环
  
- [04-TensorFlow常用函数](./04-深度学习框架/TensorFlow/04-TensorFlow常用函数.ipynb)
  
- [05-tf.function与AutoGraph](./04-深度学习框架/TensorFlow/05-tf.function与AutoGraph.ipynb)
    - tf.function 使用详解
  
- [Torch](./04-深度学习框架/Torch) ：

  - [01-PyTorch入门](./04-深度学习框架/Torch/01-PyTorch入门.ipynb)
    - Torch 基本概念，张量，CUDA张量，自动求导
    - 创建模型：Sequential 模型，nn.Module指模型
    - Torch 数据管道
    - Torch 实现线性回归，逻辑回归，CNN，RNN，残差网络，及语言模型

## [05-深度学习](./05-深度学习)

- 创建神经网络，实现图像分类与情感分类，涉及到词向量，CNN，RNN 等模型

- CNN架构，自编码器，对抗生成网络，风格迁移基本原理　等

  

## [06-自然语言处理](./06-自然语言处理)



### 基本的文本处理：

#### 涉及到分词、词表征、文档表征，原理及代码实现



[00-文本处理的基本流程](./06-自然语言处理/00-文本处理的基本流程.ipynb)

[00-文本预处理常用函数](./06-自然语言处理/00-文本预处理常用函数.ipynb)

[01-分词](./06-自然语言处理/01-分词.ipynb)

[01-编辑距离](./06-自然语言处理/01-编辑距离.ipynb)

[02-词表征与词向量](./06-自然语言处理/02-词表征与词向量.ipynb)

[03-训练词向量](./06-自然语言处理/03-训练词向量.ipynb)

[04-文档向量](./06-自然语言处理/04-文档向量.ipynb)

[04-文档向量](./06-自然语言处理/04-文档向量.py)

[05-doc2vec](./06-自然语言处理/05-doc2vec.md) 

### PageRank和TextRank  

[06-PageRank及TextRank](./06-自然语言处理/06-PageRank及TextRank.md)

### 主题模型

[09-LDA主题模型](./06-自然语言处理/09-LDA主题模型.ipynb)

### 利用神经网络实现文本分类、语言模型、语言生成

[07-keras-imdb-classification](./06-自然语言处理/07-keras-imdb-classification.py)

[08-keras-imdb-rnn](./06-自然语言处理/08-keras-imdb-rnn.py)

[10-RNN语言模型](./06-自然语言处理/10-RNN语言模型.ipynb)

[15-基于词向量和LSTM对豆瓣影评分类(TensorFlow)](./06-自然语言处理/15-基于词向量和LSTM对豆瓣影评分类(TensorFlow).ipynb)

[16-基于RNN的字符级自然语言生成](./06-自然语言处理/16-基于RNN的字符级自然语言生成.ipynb)



[39-自然语言生成](./06-自然语言处理/39-自然语言生成.ipynb)  



## 序列标注任务：

### HMM算法、CRF算法，原理及代码实现

[11-基于HMM和Viterbi算法的序列标注](./06-自然语言处理/11-基于HMM和Viterbi算法的序列标注.ipynb)

[12-BiLSTM和CRF算法的序列标注原理](./06-自然语言处理/12-BiLSTM和CRF算法的序列标注原理.ipynb)

[13-基于BiLSTM和CRF算法的命名实体识别(PyTorch)](./06-自然语言处理/13-基于BiLSTM和CRF算法的命名实体识别(PyTorch).ipynb)

[14-基于BiLSTM和CRF算法的命名实体识别(TensorFlow)](./06-自然语言处理/14-基于BiLSTM和CRF算法的命名实体识别(TensorFlow).ipynb)



## Attention机制及Transformer模型

[25-Attention机制](./06-自然语言处理/25-Attention机制.ipynb)

[26-Attention使用示例](./06-自然语言处理/26-Attention使用示例.ipynb)

[27-基于Attention的中译英(TensorFlow)](./06-自然语言处理/27-基于Attention的中译英(TensorFlow).ipynb)

[28-基于Attention的图片标注(TensorFlow)](./06-自然语言处理/28-基于Attention的图片标注(TensorFlow).ipynb)

[30-Transformer模型及源代码(PyTorch)](./06-自然语言处理/30-Transformer模型及源代码(PyTorch).ipynb)

[31-基于Transformer的中译英(TensorFlow)](./06-自然语言处理/31-基于Transformer的中译英(TensorFlow).ipynb)

[32-基于Transformer的seq2seq模型(PyTorch)](./06-自然语言处理/32-基于Transformer的seq2seq模型(PyTorch).ipynb)

[33-Transformer-XL](./06-自然语言处理/33-Transformer-XL.ipynb)

[34-Transformer优化](./06-自然语言处理/34-Transformer优化.ipynb)



## BERT及后续预训练模型

### BERT 模型原理及代码实现，基于 PyTorch 和 TensorFlow

[20-ELMo模型](./06-自然语言处理/20-ELMo模型.ipynb) 

[40-BERT基本原理及运用](./06-自然语言处理/40-BERT基本原理及运用.ipynb)

[41-BERT创建训练数据(Tensorflow)](./06-自然语言处理/41-BERT创建训练数据(Tensorflow).ipynb)

[42-BERT模型详解及代码实现(Tensorflow)](./06-自然语言处理/42-BERT模型详解及代码实现(Tensorflow).ipynb)

[42-BERT模型详解及代码实现(Tensorflow)](./06-自然语言处理/42-BERT模型详解及代码实现(Tensorflow).py)

[43-BERT模型详解及代码实现(Tensorflow)](./06-自然语言处理/43-BERT模型详解及代码实现(Tensorflow).py)

[44-BERT预训练及代码实现(Tensorflow)](./06-自然语言处理/44-BERT预训练及代码实现(Tensorflow).ipynb)



#### BERT官方源码

[40-bert-modeling(TensorFlow)](./06-自然语言处理/40-bert-modeling(TensorFlow))



### 基于BERT的自然语言处理任务



[45-基于BERT的文本分类](./06-自然语言处理/45-基于BERT的文本分类.ipynb)

[46-基于BERT的问答任务](./06-自然语言处理/46-基于BERT的问答任务.ipynb)

[47-基于BERT的文本摘要](./06-自然语言处理/47-基于BERT的文本摘要.ipynb)

[48-基于BERT的命名实体识别](./06-自然语言处理/48-基于BERT的命名实体识别.ipynb)

[49-以BERT为底层结构的分类模型](./06-自然语言处理/49-以BERT为底层结构的分类模型.ipynb) 



### BERT的优化改进及后续预训练模型

[50-BERT加速](./06-自然语言处理/50-BERT加速.ipynb)

[51-XLNet模型](./06-自然语言处理/51-XLNet模型.ipynb)

[52-ALBERT](./06-自然语言处理/52-ALBERT.ipynb)

[55-RoBERTa](./06-自然语言处理/55-RoBERTa.ipynb)

[65-GPT](./06-自然语言处理/65-GPT.ipynb)

[66-miniGPT(TensorFlow)](./06-自然语言处理/66-miniGPT(TensorFlow).ipynb)

[66-miniGPT(TensorFlow)](./06-自然语言处理/66-miniGPT(TensorFlow).py)

[67-ERNIE](./06-自然语言处理/67-ERNIE.ipynb)

[80-ELECTR预训练模型](./06-自然语言处理/80-ELECTR预训练模型.ipynb)

[90-Reformer模型](./06-自然语言处理/90-Reformer模型.ipynb)

[99-预训练模型MASK方法总结](./06-自然语言处理/99-预训练模型MASK方法总结.ipynb)

[99-预训练模型总结](./06-自然语言处理/99-预训练模型总结.ipynb)  



## 其它



[35-NLP数据增强](./06-自然语言处理/35-NLP数据增强.ipynb)

[36-生成模型的解码方法](./06-自然语言处理/36-生成模型的解码方法.ipynb)

[37-positioanl encoding](./06-自然语言处理/37-positioanl-encoding.ipynb)

[38-填充与遮盖](./06-自然语言处理/38-填充与遮盖.ipynb)









# [07-模型部署](./07-模型部署)  

- [tensorflow-serving](./07-模型部署/tensorflow-serving.ipynb)
  
- 利用 tensorflow-serving 部署 tensorflow 训练得到的模型
  
- [部署PyTorch模型](./07-模型部署/部署PyTorch模型.ipynb)

  - 部署 PyTorch 训练得到的模型

  




