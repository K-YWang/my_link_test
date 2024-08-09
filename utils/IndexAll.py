# author: Wwwy
# date: 2024/07/26
# desc: 该文件主要实现了一些常用的网络相似性指标，包括Common_neighbors, Jaccard, Salton等
# 原理的详情参考info文件夹下的指标图片

import numpy as np
from scipy.spatial.distance import pdist, squareform
import networkx as nx
from sklearn.preprocessing import Normalizer


class IndexAll:
    def __init__(self, matrix):
        self.matrix = matrix
        self.normal = Normalizer()

    # Common neighbors 指标 -- 邻接矩阵中心对称，每一行或每一列代表一个节点的邻居，矩阵运算刚好让
    # n , m 两个节点对应的行列相乘,若非共同节点最后结果为0，反之为1，最后相加即可得到共同邻居数
    def Common_neighbors(self):
        cn_sim = np.dot(self.matrix, self.matrix)
        # cn_sim = self.normal.fit_transform(cn_sim)
        return cn_sim 

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

                
        with np.errstate(divide='ignore', invalid='ignore'):
            sim = np.divide(sim, res_sim)
            sim[np.isnan(sim)] = 0  # 将NaN值替换为0

        # sim = self.normal.fit_transform(sim)
        return sim


    def sSalton(self):
        #计算Salton指标

        #计算共同邻居数
        sim = np.dot(self.matrix , self.matrix)

        res_mid = np.zeros(sim.shape)
        
        degree = np.sum(self.matrix, axis=1) # 求每个节点的度
        res_mid = np.sqrt(np.outer(degree, degree)) # 求每个节点的度的乘积的开方
                    
        with np.errstate(divide='ignore', invalid='ignore'):
            salton_similarity = np.divide(sim, res_mid)
            salton_similarity[np.isnan(salton_similarity)] = 0  # 将NaN值替换为0

        # salton_similarity = self.normal.fit_transform(salton_similarity)
        return salton_similarity
    

    def sHPI(self):
        #计算sHPI指标
        #计算共同邻居数
        sim = np.dot(self.matrix , self.matrix)

        res_mid = np.zeros(sim.shape)
        degree = np.sum(self.matrix, axis=1)
        res_mid = np.minimum(degree[:, None], degree[None, :])

        with np.errstate(divide='ignore', invalid='ignore'):
            shpi_similarity = np.divide(sim, res_mid)
            shpi_similarity[np.isnan(shpi_similarity)] = 0
        
        # shpi_similarity = self.normal.fit_transform(shpi_similarity)s
        return shpi_similarity
    

    def sHDI(self):
        #计算sHDI指标
        #计算共同邻居数
        sim = np.dot(self.matrix , self.matrix)

        res_mid = np.zeros(sim.shape)
        degree = np.sum(self.matrix, axis=1)
        res_mid = np.maximum(degree[:, None], degree[None, :])

        with np.errstate(divide='ignore', invalid='ignore'):
            shdi_similarity = np.divide(sim, res_mid)
            shdi_similarity[np.isnan(shdi_similarity)] = 0

        # shdi_similarity = self.normal.fit_transform(shdi_similarity)s
        return shdi_similarity
    

    def sLLHN(self):
        #计算sLLHN指标
        #计算共同邻居数
        sim = np.dot(self.matrix , self.matrix)

        res_mid = np.zeros(sim.shape)
        degree = np.sum(self.matrix, axis=1)
        res_mid = np.outer(degree, degree)

        with np.errstate(divide='ignore', invalid='ignore'):
            sllhn_similarity = np.divide(sim, res_mid)
            sllhn_similarity[np.isnan(sllhn_similarity)] = 0

        # sllhn_similarity = self.normal.fit_transform(sllhn_similarity)
        return sllhn_similarity
    

    def sAA(self):
        # sAA指标对AA指标进行了对称处理，使得它不仅考虑两个节点之间的共同邻居，还考虑这些邻居的稀有性，同时考虑两个节点自身的度数对相似性的影响。
        #计算sAA指标
        #计算共同邻居数
        sim = np.dot(self.matrix , self.matrix)

        degree = np.sum(self.matrix , axis = 1)
        sAA_similarity = np.zeros(sim.shape)

        for i in range(self.matrix.shape[0]):
            for j in range(i, self.matrix.shape[1]):

                common_neighbors = np.intersect1d(np.where(self.matrix[i] > 0)[0], np.where(self.matrix[j] > 0)[0])

                if(len(common_neighbors) > 0):
                     with np.errstate(divide='ignore', invalid='ignore'):
                        sAA_num = np.sum(1 / np.log(degree[common_neighbors]))
                        # sllhn_similarity[np.isnan(sllhn_similarity)] = 0
                        if np.isnan(sAA_num):
                            sAA_num = 0
                    
                    
                else:
                    sAA_num = 0
                
                sAA_similarity[i][j] = sAA_num
                sAA_similarity[j][i] = sAA_num
        
        # sAA_similarity = self.normal.fit_transform(sAA_similarity)
        return sAA_similarity
    
    def sRA(self):
        #计算sRA指标
        #计算共同邻居数
        sim = np.dot(self.matrix , self.matrix)

        degree = np.sum(self.matrix , axis = 1)
        sRA_similarity = np.zeros(sim.shape)

        for i in range(self.matrix.shape[0]):
            for j in range(i, self.matrix.shape[1]):

                common_neighbors = np.intersect1d(np.where(self.matrix[i] == 1)[0], np.where(self.matrix[j] == 1)[0])

                if(len(common_neighbors) > 0):
                    sRA_num = np.sum(1 / degree[common_neighbors])

                else:
                    sRA_num = 0
                
                sRA_similarity[i][j] = sRA_num
                sRA_similarity[j][i] = sRA_num
        
        # sRA_similarity = self.normal.fit_transform(sRA_similarity)
        return sRA_similarity
    

    def sPA(self):
        # sPA指标对PA指标进行了对称处理，使得它不仅考虑两个节点之间的共同邻居，还考虑这些邻居的稀有性，同时考虑两个节点自身的度数对相似性的影响。
        #计算sPA指标
        #计算共同邻居数
        sim = np.dot(self.matrix , self.matrix)

        degree = np.sum(self.matrix , axis = 1)
        sPA_similarity = np.outer(degree, degree)

        # sPA_similarity = self.normal.fit_transform(sPA_similarity)
        return sPA_similarity
    

    def sRandomWalk(self, C=0.85, max_iter=100, tol=1e-6):
        num_nodes = self.matrix.shape[0]
        P = np.zeros_like(self.matrix, dtype=float)
        
        # 构建转移概率矩阵P
        for i in range(num_nodes):
            row_sum = np.sum(self.matrix[i])
            if row_sum > 0:
                P[i] = self.matrix[i] / row_sum
        
        # print(P)

        # 初始化相似度矩阵
        sim = np.eye(num_nodes)
        
        # 迭代计算相似度矩阵
        for _ in range(max_iter):
            new_sim = C * np.dot(P, sim) + (1 - C) * np.eye(num_nodes)
            
            # 检查收敛
            if np.linalg.norm(new_sim - sim) < tol:
                break
            
            sim = new_sim
        
        # sim = self.normal.fit_transform(sim)
        return sim
    
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
        
        # cndp_sim = self.normal.fit_transform(cndp_sim)
        return cndp_sim