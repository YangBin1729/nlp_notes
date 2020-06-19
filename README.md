# 自然语言处理学习笔记

机器学习及深度学习原理和示例，基于 Tensorflow 和 PyTorch 框架，Transformer、BERT、ALBERT等最新预训练模型及源代码详解，及基于预训练模型进行各种自然语言处理任务。以及模型部署



[01-传统模型](./01-传统模型) 

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
          

[02-机器学习](./02-机器学习)

- 机器学习算法，及其应用

  

[03-神经网络Python实现](./03-神经网络Python实现)

- python 实现基本的神经网络：激活函数，损失函数，前向传播，反向传播
- python 实现各种梯度下降算法，初始化，Batch Normalization，正则化
- python 实行 CNN 



[04-深度学习框架](./04-深度学习框架)

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

[05-深度学习](./05-深度学习)

- 创建神经网络，实现图像分类与情感分类，涉及到词向量，CNN，RNN 等模型

- CNN架构，自编码器，对抗生成网络，风格迁移基本原理　等

  

[06-自然语言处理](./06-自然语言处理)

- 涉及文本处理、词向量与文档向量，神经网络处理NLP任务，序列标注，注意力机制与Transformer，BERT及后续预训练模型 等

  

[07-模型部署](./07-模型部署)  

- [tensorflow-serving](./07-模型部署/tensorflow-serving.ipynb)
  
- 利用 tensorflow-serving 部署 tensorflow 训练得到的模型
  
- [部署PyTorch模型](./07-模型部署/部署PyTorch模型.ipynb)

  - 部署 PyTorch 训练得到的模型

  




