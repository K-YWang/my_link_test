from utils.NewIndex import NewIndex as Ni
import networkx as nx

G = nx.read_adjlist("data/linklist.txt", nodetype=int)

adjacent_matrix = nx.to_numpy_array(G)

Tindex = Ni(adjacent_matrix)

cndp_res = Tindex.CNDP()

print(adjacent_matrix)
print(cndp_res)