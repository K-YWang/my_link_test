import numpy as np
from scipy.spatial.distance import pdist, squareform

class IndexAll:
    def __init__(self, matrix):
        self.matrix = matrix

    # Common neighbors 指标 -- 邻接矩阵中心对称，每一行或每一列代表一个节点的邻居，矩阵运算刚好让
    # n , m 两个节点对应的行列相乘,若非共同节点最后结果为0，反之为1，最后相加即可得到共同邻居数
    def Common_neighbors(self):
        return np.dot(self.matrix, self.matrix)   # 
    

    def Jaccard(self):
        #计算Jaccard指标
        #matrix-网络的邻接矩阵，sim-相似性矩阵
        sim = np.dot(self.matrix , self.matrix)

        res_sim = np.zeros(sim.shape)

        for i in range(self.matrix.shape[0]):
            for j in range(i, self.matrix.shape[1]):
                row = self.matrix[i]
                col = self.matrix[: , j]

                # 对行和列进行逻辑或运算
                logical_or_result = np.logical_or(row, col)

                # 计算逻辑与的个数
                logical_or_count = np.sum(logical_or_result)
                res_sim[i][j] = logical_or_count
                res_sim[j][i] = logical_or_count

                
        sim = np.divide(sim, res_sim)

            # 将邻接矩阵转换为布尔类型
        # bool_matrix = self.matrix.astype(bool)
        # print(bool_matrix)
        # # 计算Jaccard距离，并转换为相似性
        # jaccard_distances = pdist(bool_matrix, metric='jaccard')
        # print(jaccard_distances)

        # jaccard_similarities = 1 - squareform(jaccard_distances)
        # print(jaccard_similarities)
        
        # return jaccard_similarities

        # 将邻接矩阵转换为布尔类型
        bool_matrix = self.matrix.astype(bool)

        # 计算交集
        intersection = np.dot(self.matrix, self.matrix)

        # 计算并集
        union = np.bitwise_or(bool_matrix[:, None], bool_matrix[None, :]).sum(axis=0)

        # print(union)

        # 计算Jaccard相似性
        jaccard_similarities = intersection / union

        return jaccard_similarities


    def Adamic_Adar(self):
        pass