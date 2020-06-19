### PageRank算法

**问题**：如何对相互链接的巨量网页进行重要性排序？

**PageRank算法**：

- 一个网页被其它很多网页链接，该网页 PageRank 值会较高 
- 一个网页被被一个 PageRank 值高的网页链接，该网页的 PageRank 值会因此而提高 

**示例**：

四个网页，其相互链接关系如下图所示：

<img src="images/PageRank.png" style="zoom:48%;" />

基于上图，计算用户在某网页时，跳转到其它网站的概率，得到如下转移矩阵 $M$ 。如当用户在 A 网页时，则用户跳转到 B、C、D 网页的概率各有 1/3。

| 网页 |  A   |  B   |  C   |  D   |
| :--: | :--: | :--: | :--: | :--: |
|  A   |  0   | 1/2  |  1   |  0   |
|  B   | 1/3  |  0   |  0   | 1/2  |
|  C   | 1/3  |  0   |  0   | 1/2  |
|  D   | 1/3  | 1/2  |  0   |  0   |

所有网页的 PageRank 向量为 $V_{(4,)}$ 。假设用户在每个网页的概率是相等的，都为 $1/n$ ，初始化 $V_0$。$V_1 = MV_0$ ，不断迭代计算 $V_i = MV_{i-1}$ ，直到 $|V_i-V_{i-1}|<\epsilon$ ，即 $V$ 收敛到一个恒定值，即为最终的PageRank向量。

假设查看当前网页的概率是 *a*, 从地址栏跳转的概率是1−*α*。则 $V_i = \alpha\cdot MV_{i-1}+(1-\alpha)\cdot V_0$ 。

网页 A 的 PageRank 的值可以理解【**每个网页跳转到 A 的概率乘以该网页的 PageRank 值**】：$B\rightarrow A$ ，$C\rightarrow A$ ，$D\rightarrow A$ 的概率分别为 0、1/2、1、0；初始化时，网页 B、C、D 的 PageRank 值分别为 1/4

```python
>>> nodes = 'ABCD'
>>> edges = [('A', 'B'), ('A', 'C'), ('A', 'D'),
             ('B', 'A'), ('B', 'D'),
             ('C', 'A'), ('D', 'B'), ('D', 'C')]
>>> G = nx.DiGraph()
>>> G.add_nodes_from(nodes)
>>> G.add_edges_from(edges)
>>> pr = nx.pagerank(G, alpha=0.9)
{'A': 0.3245609358176831, 'B': 0.22514635472743894, 'C': 0.22514635472743894, 'D': 0.22514635472743894}
```





## 基于 PageRank 计算单词的重要性

```python
>>> from gensim.summarization.keywords import get_graph
>>> from gensim.summarization.pagerank_weighted import pagerank_weighted
>>> graph = get_graph("The road to hell is paved with good intentions.")
>>> result = pagerank_weighted(graph)
{'road': 0.051128871128006126, 'hell': 0.051128871128006154, 'pave': 0.05112887112800624, 'good': 0.7043285865317153, 'intent': 0.7043285865317152}
```

todo：如何用于中文

todo：背后的语法细节

- [ ] jieba.analyse.textrank 完整源码



### TextRank 计算句子的重要性

$$
\begin{align}
&S(v_i) = (1-d) + d\sum\frac{w_{ji}}{\sum{w_{jk}}}S(v_j)\\
&j\in\text{所有能指向节点 $i$ 的节点};\ \ k\in\text{节点 $j$ 能指向的所有节点}
\end{align}
$$

- 右侧的求和表示每个相邻句子对本句子的贡献程度；与提取关键字的时候不同，一般认为全部句子都是相邻的，不再提取窗口。
- 

- 

$$
PR(v_i) = (1-d) + d\cdot\sum\frac{s_{ji}}{\sum{s_{jk}}}PR(v_j)\\
$$

$PR(v_i)$：第 $i$ 个句子的 PageRank 值；

$PR(v_j)$：第 $j$ 个句子的 PageRank 值；

$s_{ji}$：第 $j$ 个句子和第 $i$ 个句子的相似度

下述推导是基于 M 为对称矩阵：
$$
M = 
\left[
\begin{matrix}
 s_{11}  & s_{12}  & \cdots & s_{1n}\\
 s_{21}  & s_{22}  & \cdots & s_{2n}\\
 \vdots & \vdots & \ddots & \vdots \\
 s_{n1}  & s_{n2}  & \cdots & s_{nn} \\
\end{matrix}
\right]\quad\quad 
R_i^t = (1-d) + d\cdot\sum_{j=1}^{n}\frac{s_{ji}}{\sum_{k=1}^{n}{s_{jk}}}R_j^{t-1}\\
\quad\\
\quad\\
\begin{align}
\sum_{j=1}^{n}\frac{s_{ji}}{\sum_{k=1}^{n}{s_{jk}}}R_j^{t-1}
&=s_{1i}\cdot\frac{R_1^{t-1}}{sum(M[1,:])}+s_{2i}\cdot\frac{R_1^{t-1}}{sum(M[2,:])}+\cdots+s_{ni}\cdot\frac{R_n^{t-1}}{sum(M[n,:])}\\
&=
  \bigg[
  \begin{array}{ccc}
   s_{1i} & s_{2i} &\cdots & s_{ni}
  \end{array}
  \bigg]
  
 
  \begin{bmatrix}
   \frac{R_1^{t-1}}{sum(M[1,:])} &\\ \frac{R_2^{t-1}}{sum(M[2,:])} &\\ \vdots &\\ \frac{R_n^{t-1}}{sum(M[n,:])}
  \end{bmatrix}\\
  
&=M[:,i]^T\cdot\frac{R^{t-1}}{sum(M, axis=1)}
\quad\\
\quad\\ 
R^t &= (1-d) + d\cdot M^T\cdot\frac{R^{t-1}}{sum(M, axis=1)} 
\end{align}
$$

### 基于TextRank 进行文本摘要

- 将文本分句，
- 再将每个句子分词，利用词向量生成该句的句子向量
- 计算句子间的余弦相似度矩阵
- 相似度超过阈值的两个句子，语义可以关联起来，边的权重即为相似度
- 句子权重计算：根据公式，迭代传播权重计算各句子的得分；
- 抽取文摘句：将句子得分进行倒序排序，抽取重要度最高的T个句子作为候选文摘句
- 形成文摘：根据字数或句子数要求，从候选文摘句中抽取句子组成文摘。