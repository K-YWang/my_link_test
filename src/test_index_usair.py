from utils.NewIndex import NewIndex as Ni
from utils.IndexAll import IndexAll as Ia
from utils.ReadFile import ReadFile as Rff
from utils.roc_auc import roc_auc as Ra

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, roc_auc_score

file_path = "data/data_set/USair.txt"
rf = Rff(file_path)
vertex_dict, edges_list = rf.readUSair()
vertex_num = len(vertex_dict)

# 将二维数组写入CSV文件
# np.savetxt('data/show_res.txt', adjacent_matrix, delimiter=',', fmt='%lf')

# 创建一个无向图, 用于存储USair数据集,邻接矩阵
all_matrix = np.zeros((len(vertex_dict) + 1, len(vertex_dict) + 1))
for edge in edges_list:
    node1, node2, weight = edge
    all_matrix[node1][node2] = 1
    all_matrix[node2][node1] = 1

train_data , test_data = train_test_split(edges_list, test_size=0.2, random_state=42)
# 使用集合存储训练集边
train_data_set = set((min(edge[0], edge[1]), max(edge[0], edge[1])) for edge in train_data)
test_data_set = set((min(edge[0], edge[1]), max(edge[0], edge[1])) for edge in test_data)


# 创建一个无向图, 用于存储USair训练数据集,邻接矩阵
adjacent_matrix = np.zeros((len(vertex_dict) + 1, len(vertex_dict) + 1))
for edge in train_data:
    node1, node2, weight = edge
    adjacent_matrix[node1][node2] = 1
    adjacent_matrix[node2][node1] = 1

# 实例化一个NewIndex对象，并训练sAA指标
Tindex = Ia(adjacent_matrix)
Nindex = Ni(adjacent_matrix)


sAA_sim = Nindex.CNDP(1)


# 计算得分
scores = []
for i in range(1 , vertex_num + 1):
    for j in range(i + 1 , vertex_num + 1):
        if (i , j) not in train_data_set and (j , i) not in train_data_set:
           scores.append([i , j , sAA_sim[i][j]])

# 按照得分排序
scores.sort(key = lambda x : x[2] , reverse = True)

# 计算得分
y_true = []
y_scores = []

for node1, node2, score in scores:
    y_scores.append(score)

    if (node1, node2) in test_data_set or (node2, node1) in test_data_set:
        y_true.append(1)
    else:
        y_true.append(0)

np.savetxt('data/show_res.txt', scores , delimiter=',', fmt='%d')
precision, recall, _ = precision_recall_curve(y_true, y_scores)
auc_roc = roc_auc_score(y_true, y_scores)

Score = Ra(y_true, y_scores)
Score.get_roc()

print("auc_scores:" , Score.get_auc())


# print(f"precision: {precision}")
# print(f"recall: {recall}")
print(f"AUC-ROC: {auc_roc}")