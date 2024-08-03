# author: Wwwy
# date: 2024/07/26
# desc: 该文件主要实现了一些新的网络相似性指标
import numpy as np
from scipy.spatial.distance import pdist, squareform
from itertools import combinations

class NewIndex:
    def __init__(self, matrix):
        self.matrix = matrix
    
    def CNDP(self):
        num_nodes = self.matrix.shape[0]
        sim = np.dot(self.matrix, self.matrix) # 计算共同邻居数

        cndp_res = np.zeros(sim.shape)

        cndp_score = 0.0

        for i in range(num_nodes):
            for j in range(i + 1 , num_nodes):
                common_neighbors = np.intersect1d(np.where(self.matrix[i] == 1)[0], np.where(self.matrix[j] == 1)[0])

                if len(common_neighbors) >= 2:
                     # 计算共同邻居之间的实际连接数
                    actual_edges = 0
                    for u, v in combinations(common_neighbors, 2):
                        if self.matrix[u, v] == 1:
                            actual_edges += 1

                    # 计算可能的最大连接数
                    max_possible_edges = len(common_neighbors) * (len(common_neighbors) - 1) / 2

                    # 计算CNDP分数
                    if max_possible_edges > 0:
                        cndp_score = actual_edges / max_possible_edges
                    else:
                        cndp_score = 0

                else:
                    cndp_score = 0
                

                cndp_res[i][j] = cndp_score
                cndp_res[j][i] = cndp_score   # 对称

        return cndp_res


