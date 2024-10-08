import numpy as np

def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A

def normalize_digraph(A):  
    Dl = np.sum(A, 0)
    h, w = A.shape
    Dn = np.zeros((w, w))
    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD

def get_spatial_graph(num_node, self_link, inward, outward): #
    I = edge2mat(self_link, num_node)
    In = normalize_digraph(edge2mat(inward, num_node))
    Out = normalize_digraph(edge2mat(outward, num_node))
    A = np.stack((I, In, Out)) #自身，in，out的临界矩阵堆叠
    return A

def k_adjacency(A, k, with_self=False, self_factor=1):
    assert isinstance(A, np.ndarray)
    I = np.eye(len(A), dtype=A.dtype) #对角阵对应于根节点
    if k == 0:
        return I
    Ak = np.minimum(np.linalg.matrix_power(A + I, k), 1) \
       - np.minimum(np.linalg.matrix_power(A + I, k - 1), 1) #k跳的图-(k-1)跳的图 我经过推倒，发现就是A^k
    if with_self:
        Ak += (self_factor * I) #加上根节点
    return Ak

def normalize_adjacency_matrix(A): #归一化
    node_degrees = A.sum(-1) #D
    degs_inv_sqrt = np.power(node_degrees, -0.5) #D^1/2
    norm_degs_matrix = np.eye(len(node_degrees)) * degs_inv_sqrt #这才是正宗的对角阵
    return (norm_degs_matrix @ A @ norm_degs_matrix).astype(np.float32) #D^1/2AD^1/2

def get_adjacency_matrix(edges, num_nodes=25):
    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for edge in edges:
        A[edge] = 1.
    return A