import sys
import re

class ReadFile:
    def __init__(self):
        pass
    def readUSair(self,file_path):
        
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
    
    def readnet(self , file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
        nodes = []
        edges = []
        reading_nodes = False
        reading_edges = False
        
        for line in lines:
            line = line.strip()

            if line.startswith('*vertices'):
                reading_nodes = True
                reading_edges = False
                continue
            elif line.startswith('*edges') or line.startswith('*arcs'):
                reading_nodes = False
                reading_edges = True
                continue
            

            if reading_nodes:
                parts = line.split()
                node_id = int(parts[0])
                node_label = parts[1].strip('"')
                nodes.append((node_id, node_label))
            
            if reading_edges:
                parts = line.split()
                node1 = int(parts[0])
                node2 = int(parts[1])
                weight = float(parts[2]) if len(parts) > 2 else 1.0
                edges.append((node1, node2, weight))
        
        return nodes, edges
