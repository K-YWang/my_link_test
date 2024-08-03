from utils.NewIndex import NewIndex as Ni
from utils.IndexAll import IndexAll as Ia

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
def parse_usair_file(file_path):
    # 用于存储点ID与名字的对应关系
    vertex_dict = {}
    
    # 用于存储边的信息
    edges_list = []

    # 标志位，用于判断当前读到的部分是顶点还是边
    parsing_vertices = False
    parsing_edges = False

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            if line.startswith('*Vertices'):
                # 开始解析顶点部分
                parsing_vertices = True
                parsing_edges = False
                continue
            elif line.startswith('*Edges'):
                # 开始解析边部分
                parsing_vertices = False
                parsing_edges = True
                continue

            elif line.startswith('*Arcs'):
                continue

            if parsing_vertices:
                # 解析顶点，格式：ID "Name" X Y Z
                # 使用正则表达式解析行内容
                match = re.match(r'(\d+)\s+"([^"]+)"\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', line)
                if match:
                    vertex_id = int(match.group(1))
                    vertex_name = match.group(2)
                    # 解析坐标值（可以忽略或根据需要存储）
                    x_coord = float(match.group(3))
                    y_coord = float(match.group(4))
                    z_coord = float(match.group(5))
                    vertex_dict[vertex_id] = vertex_name

            elif parsing_edges:
                # 解析边，格式：ID1 ID2 Weight
                parts = line.split()
                node1 = int(parts[0])
                node2 = int(parts[1])
                weight = float(parts[2])
                edges_list.append((node1, node2, weight))


    return vertex_dict, edges_list

file_path = "data/USair.txt"
vertex_dict, edges_list = parse_usair_file(file_path)

# 创建一个无向图, 用于存储USair数据集,邻接矩阵
adjacent_matrix = np.zeros((len(vertex_dict) + 1, len(vertex_dict) + 1))
for edge in edges_list:
    node1, node2, weight = edge
    adjacent_matrix[node1][node2] = weight
    adjacent_matrix[node2][node1] = weight



# 将二维数组写入CSV文件
# np.savetxt('data/show_res.txt', adjacent_matrix, delimiter=',', fmt='%lf')

Tindex = Ia(adjacent_matrix)
sAA_sim = Tindex.sAA()

np.savetxt('data/show_res.txt', sAA_sim , delimiter=',', fmt='%lf')