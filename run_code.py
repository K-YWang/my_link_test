# author: Wwwy
# date: 2024/08/02
import os
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
import inspect

file_name_list = ["BUP" , "ADV" , "CDM" , "CEG" , "CGS" , 
                  "EML" , "ERD" , "FBK" , "GRQ" , "HMT" , 
                  "HPD" , "HTC" , "INF" , "KHN" , "LDG" , 
                  "NSC" , "PGP" , "SMG" , "UAL" , "UPG" , 
                  "YST" , "ZWL"]

base_path = "data/datasets"
test_list = ["UAL" , "BUP"]

output_path = "data/show_res.txt"

scores_all = {}
with open(output_path , "w") as file:
    pass

with open(output_path , "a") as file:
    for dataset in test_list:
        m = {}
        m_score = {}

        for i in range(0 , 5):
            train_file = os.path.join(base_path , f"{dataset}_train_{i}.net")
            test_file = os.path.join(base_path , f"{dataset}_test_{i}.net")

            # print(train_file)
            # print(test_file)

            if os.path.exists(train_file):
                train_vertex_dict , train_edges_list = Rff().readnet(train_file)
            else:
                # print("open file failed")
                file.write("open file failed\n")

            if os.path.exists(test_file):
                test_vertex_dict , test_edges_list = Rff().readnet(test_file)
            else:
                # print("open file failed")
                file.write("open file failed\n")

            
            # 构建邻接矩阵
            train_matrix = np.zeros((len(train_vertex_dict) , len(train_vertex_dict)))
            for edge in train_edges_list:
                train_matrix[edge[0] - 1][edge[1] - 1] = 1
                train_matrix[edge[1] - 1][edge[0] - 1] = 1

            test_matrix = np.zeros((len(test_vertex_dict) , len(test_vertex_dict)))
            for edge in test_edges_list:
                test_matrix[edge[0] - 1][edge[1] - 1] = 1
                test_matrix[edge[1] - 1][edge[0] - 1] = 1
            
            # 计算指标
            Nindex = Ni(train_matrix)
            # sim = Nindex.CNDP(1)
            Tindex = Ia(train_matrix)
            methods = inspect.getmembers(Tindex, predicate=inspect.ismethod)

            for name , method in methods:
                if(name == "__init__"):
                    continue

                sim = method()
                # print(f"Running {name} on {dataset}")
                file.write(f"Running {name} on {dataset}\n")

                scores = []

                for i in range(0 , len(train_vertex_dict)):
                    for j in range(i + 1 , len(train_vertex_dict)):
                        if train_matrix[i][j] != 1:
                            scores.append([i , j , sim[i][j]])

                # 按照得分排序
                scores.sort(key = lambda x : x[2] , reverse = True)

                y_true = []
                y_scores = []

                for node1, node2, score in scores:
                    y_scores.append(score)

                    if test_matrix[node1][node2] == 1:
                        y_true.append(1)
                    else:
                        y_true.append(0)

                # 计算roc_auc
                precision, recall, _ = precision_recall_curve(y_true, y_scores)
                auc_roc = roc_auc_score(y_true, y_scores)

                # print(f"dataset: {dataset} , auc_roc: {auc_roc}")
                # print('\n')

                file.write(f"dataset: {dataset} , auc_roc: {auc_roc}\n\n")

                if name not in m:
                    m[name] = []
                m[name].append(auc_roc)

            # print(f"success - {dataset}")
            file.write(f"success - {dataset}\n")
        print(f"done - {dataset}")

        #一个数据集结束
        for name , l in m.items():
            m_score[name] = np.mean(l)
        
        scores_all[dataset] = m_score
    
    # 所有数据集结束
    df = pd.DataFrame(scores_all)
    df = df.T
    print(df)

    print("done")
    file.write("done\n")
