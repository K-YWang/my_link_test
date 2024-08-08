# author: Wwwy
# date: 2024/07/26
# desc: 该文件主要实现了一些新的网络相似性指标
import numpy as np
from scipy.spatial.distance import pdist, squareform
from itertools import combinations
import networkx as nx


class NewIndex:
    def __init__(self, matrix):
        self.matrix = matrix
    
    def CNDP(self , bate):
        num_nodes = self.matrix.shape[0]
        cndp_sim = np.zeros((num_nodes + 1, num_nodes + 1))

        # 计算聚类系数
        G = nx.from_numpy_array(self.matrix)
        cluster_coefficient = nx.clustering(G)

        # print(f"cluster_coefficient: {cluster_coefficient}")
        average_cluster_coefficient = np.mean(list(cluster_coefficient.values()))

        for i in range(1 , num_nodes):
            for j in range(i + 1 , num_nodes):
                common_neighbors = np.where((self.matrix[i] > 0) & (self.matrix[j] > 0))[0]
                CNDP_value = 0
                # print(f"{i} , {j} , common_neighbors: {common_neighbors}")

                if len(common_neighbors) == 0:
                    CNDP_value = 0
                else:
                    for node in common_neighbors:
                        neighbors = np.where(self.matrix[node] > 0)[0]

                        intersection = set(set(neighbors) | set(common_neighbors))

                        # print(f"node: {node} , neighbors: {neighbors} , intersection: {intersection}")

                        CNDP_value += len(intersection) * (len(neighbors) ** (-bate * average_cluster_coefficient))
                
                cndp_sim[i][j] = CNDP_value
                cndp_sim[j][i] = CNDP_value
        
        return cndp_sim
    
                        


