预测句子中下一个单词的任务，词向量是该任务的一个非直接的结果，随机初始化，但最终捕获语义。

paragraph vectors，同样参与预测下一个单词的任务，给定从段落中采样的许多上下文

## 一、

矩阵D，每一列表示一个段落向量；矩阵W，每一列表示一个词向量。两者平均或联结起来，预测上下文中的下一个单词。paragraph token可以被认为是另一个单词，作用是记忆从当前上下文缺失的信息，或段落的主题。因此也被称为Distributed Memory Model of Paragraph Vectors (PV-DM).

<img src="images/Distributed Representations of Sentences and Documents.jpg" style="zoom:50%;" />

上下文是固定长度，并通过一个滑动窗口遍历段落采样；paragraph vector只在同一个段落内的不同上下文间共享，不同段落各自的向量；词向量是全文共享。

预测时，需要进行 inference 步骤， 计算新段落的段落向量，而所有的词向量W和权重固定

N个段落，段落向量p维，词汇表M个单词，词向量维度q，除去softmax参数，总共 $N\times p + M\times q$ 个参数。随机梯度下降训练时，参数更新稀疏而高效的。

两个阶段：

- 在语料上训练词向量W、softmax权重矩阵U，b、段落向量D
- 新语料，将其段落向量添加到D中，保持W、U、b固定。D就可以用来预测，例如逻辑回归分类器分类文本

段落向量优点：

- 从无标签的数据上学习到的，适用于没有足量标签数据的任务
- 弥补了词袋模型的缺点：首先继承了词向量的语义，其次至少考虑了较小上下文中的语序，如 n-gram 模型一样，但维持较低的维度

## 二、

忽略上下文，但强迫模型预测段落中随机采样的单词。Distributed Bag of Words version of Paragraph Vector (PV-DBOW)

<img src="images/Distributed Bag of Words.jpg" style="zoom:50%;" />

模型只需要储存softmax权重，不需要储存词向量