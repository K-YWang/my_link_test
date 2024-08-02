# index all

## CNDP（Common Neighbor Density-based Probability）指标

### 概念

**CNDP**（Common Neighbor Density-based Probability）是一种用于链路预测的新型指标，专注于通过共同邻居评估网络中两个节点连接的可能性。CNDP不仅考虑共同邻居的数量，还考虑这些共同邻居之间的连接密度。

### 背景

在网络中，两个节点拥有多个共同邻居通常意味着它们有更高的潜力形成连接。传统的共同邻居（Common Neighbors, CN）指标只关注共同邻居的数量：

\[ \text{CN}(A, B) = |N(A) \cap N(B)| \]

然而，CNDP在此基础上进一步考虑共同邻居间的连接密度。若共同邻居彼此间高度连接，则更有可能预示该节点对的连接潜力。

### 指标定义

CNDP的公式定义如下：

\[ \text{CNDP}(A, B) = \frac{|E(N(A) \cap N(B))|}{\binom{|N(A) \cap N(B)|}{2}} \]

其中：

- \( N(A) \) 和 \( N(B) \) 是节点 \( A \) 和节点 \( B \) 的邻居集合。
- \( |N(A) \cap N(B)| \) 表示节点 \( A \) 和节点 \( B \) 的共同邻居数量。
- \( |E(N(A) \cap N(B))| \) 表示共同邻居之间实际存在的边的数量。
- \(\binom{|N(A) \cap N(B)|}{2}\) 是共同邻居节点中可能形成的最大边数（完全图边数）。

CNDP通过对共同邻居之间的密度进行量化，评估节点对形成连接的潜在可能性。