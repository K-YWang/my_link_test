# author: Wwwy
# date: 2024/07/26
# desc: 该文件主要实现了一些常用的网络相似性指标，包括Common_neighbors, Jaccard, Salton等
# 原理的详情参考info文件夹下的指标图片

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

                
        with np.errstate(divide='ignore', invalid='ignore'):
            sim = np.divide(sim, res_sim)
            sim[np.isnan(sim)] = 0  # 将NaN值替换为0

        return sim
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
        # bool_matrix = self.matrix.astype(bool)

        # # 计算交集
        # intersection = np.dot(self.matrix, self.matrix)

        # # 计算并集
        # union = np.bitwise_or(bool_matrix[:, None], bool_matrix[None, :]).sum(axis=0)

        # # print(union)

        # # 计算Jaccard相似性
        # jaccard_similarities = intersection / union

        # return jaccard_similarities


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
                    sAA_num = np.sum(1 / np.log(degree[common_neighbors]))
                    
                else:
                    sAA_num = 0
                
                sAA_similarity[i][j] = sAA_num
                sAA_similarity[j][i] = sAA_num
        
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
        
        return sRA_similarity
    

    def sPA(self):
        # sPA指标对PA指标进行了对称处理，使得它不仅考虑两个节点之间的共同邻居，还考虑这些邻居的稀有性，同时考虑两个节点自身的度数对相似性的影响。
        #计算sPA指标
        #计算共同邻居数
        sim = np.dot(self.matrix , self.matrix)

        degree = np.sum(self.matrix , axis = 1)
        sPA_similarity = np.outer(degree, degree)

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
        
        return sim