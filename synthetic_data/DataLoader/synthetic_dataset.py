import logging

import igraph as ig
import networkx as nx
import numpy as np
import math
from scipy.special import expit as sigmoid
from random import choice
import pandas as pd

def is_dag(B):
    """Check whether B corresponds to a DAG.
    Args:
        B (numpy.ndarray): [d, d] binary or weighted matrix.
    """
    return nx.is_directed_acyclic_graph(nx.DiGraph(B))

#Data Simulation self.X_true = DAG_simulation.simulate_linear_sem(self.B, self.n, self.noise_type, self.rs)
def simulate_data(dv_c_ls, B, n, nn, num_of_admissible=0, noise_type=0):
    """Simulate samples from linear SEM with specified type of noise.
    Args:
        B (numpy.ndarray): [d, d] weighted adjacency matrix of DAG.
        n (int): Number of samples.
        noise_type ('gaussian', 'exponential', 'gumbel'): Type of noise.
        v_noise (float): The variance of noise
    Returns:
        numpy.ndarray: [n, d] data matrix.
    """
    def _simulate_single_equation(X, B_i, N_i, ffun_type, gfun_type):
        """Simulate samples from SEM for the i-th node.
        Args:
            X (numpy.ndarray): [n, number of parents] data matrix.
            B_i (numpy.ndarray): [d,] weighted vector for the i-th node.
            N_i (numpy.ndarray): [n,] noise vector for the i-th node.
        Returns:
            numpy.ndarray: [n,] data matrix.
        """
        # print("gfun_type = {}, B_i={} ".format(gfun_type, B_i))
        # """" X_i = g_i(f_i(PA_i))+E_i """
        # f() function type
        if ffun_type == 0:
            X_i = X
        elif ffun_type == 1:
            X_i = sigmoid(X)
        elif ffun_type == 2:
            X_i = np.sqrt(np.abs(X))
        elif ffun_type == 3:
            X_i = np.log(X)
        elif ffun_type == 4:
            X_i = np.tanh(X)

        # g() function type
        if gfun_type == 0: # Linear
            X_i = X_i @ B_i + N_i
        elif gfun_type == 1: #
            X_i = np.tanh(X_i @ B_i) + np.cos(X_i @ B_i) + np.sin(X_i @ B_i) + N_i
        elif gfun_type == 2:
            X_i = sigmoid(X_i @ B_i)**2 + N_i
        elif gfun_type == 3: #
            X_i = np.sqrt(np.abs(X_i @ B_i)) + N_i
        # elif gfun_type == 3: #
        #     X_i = np.abs(X_i @ B_i) + N_i
        return X_i

    def _generate_noise(noise_type=0, size=n):
        v_noise = 1.0
        if noise_type == 0: # Gaussian noise with equal variances
            # v_noise = np.random.uniform(low=0.5, high=6, size=1)
            N_i = np.random.normal(scale=v_noise, size=size)
        # elif noise_type == 1:
        #     v_noise = np.random.uniform(low=0.5, high=1.5, size=1)
        #     N_i = np.random.standard_t(df=5, size=n) * np.sqrt(v_noise)
        # elif noise_type == 2:
        #     v_noise = np.random.uniform(low=0.4, high=0.7, size=1)
        #     N_i = np.random.logistic(loc=0, scale=v_noise, size=n)
        # elif noise_type == 3:
        #     v_noise = np.random.uniform(low=1.2, high=2.1, size=1)
        #     N_i = np.random.uniform(low=-v_noise, high=v_noise, size=n)
        elif noise_type == 1: # Exponential noise
            N_i = np.random.exponential(scale=v_noise, size=size)
        elif noise_type == 2:  # Gumbel noise
            N_i = np.random.gumbel(scale=v_noise, size=size)
        return N_i


    d = B.shape[0]
    X = np.zeros([n, d])
    X_cf = np.zeros([n, d])
    noise_matrix = np.zeros([n, d])
    X0 = np.zeros([nn, d])
    X1 = np.zeros([nn, d])
    G = nx.DiGraph(B)
    ordered_vertices = list(nx.topological_sort(G))
    # print(ordered_vertices)
    # dv_ls = [ordered_vertices[0]]
    # dv_ls = np.random.choice(ordered_vertices[:math.ceil(d/2)], size=1).tolist()
    #dv_ls = np.random.choice(ordered_vertices[math.floor(d/10):math.ceil(d/5)], size=1).tolist()
    dv_ls = np.random.choice(ordered_vertices[:-1], size=1).tolist() # outcome=ordered_vertices[-1]
    assert num_of_admissible >= 0 & num_of_admissible <= d-2
    admissible_vars = np.random.choice(np.array(list(set(ordered_vertices[:-1]).difference(dv_ls))), size=num_of_admissible, replace=False).tolist()
    #print(dv_ls[0])
    #B[dv_ls[0], :] = B[dv_ls[0], :] * (1+np.random.uniform(0, 0.5, size=d))
    assert len(ordered_vertices) == d
    for i in ordered_vertices:
        parents = list(G.predecessors(i))
        ffun_type = 0
        # ffun_type = np.random.randint(5, size=1)[0]
        # print("Node {}: ".format(i))
        gfun_type = 0 # linear
        # gfun_type = np.random.randint(4, size=1)[0] # non-linear
        # noise_type = np.random.randint(3, size=1)[0]
        noise_type = 0 # Gaussian noise
        noise_matrix[:, i] = _generate_noise(noise_type, n)
        X_temp = _simulate_single_equation(X[:, parents], B[parents, i], noise_matrix[:, i], ffun_type, gfun_type)
        if i in admissible_vars: # intervene the admissible variables to be the mean of that observational variable.
            X_temp0 = np.full((nn,), np.mean(X_temp))
            X_temp1 = np.copy(X_temp0)
        else:
            X_temp0 = _simulate_single_equation(X0[:, parents], B[parents, i], _generate_noise(noise_type, nn), ffun_type, gfun_type)
            X_temp1 = _simulate_single_equation(X1[:, parents], B[parents, i], _generate_noise(noise_type, nn), ffun_type, gfun_type)
        if i in dv_ls:
            cla = dv_c_ls[dv_ls.index(i)]
            step = int(len(X_temp)/cla)
            sorted_id = sorted(range(len(X_temp)), key=lambda m: X_temp[m])
            for k in range(cla):
                X[sorted_id[k*step:(k+1)*step], i] = k
            X_cf[:, i] = np.ones(n) * (cla-1) - X[:, i]
            X0[:, i] = np.zeros(nn)
            X1[:, i] = np.ones(nn)
        else:
            X[:, i] = X_temp
            X_cf[:, i] = _simulate_single_equation(X_cf[:, parents], B[parents, i], noise_matrix[:, i], ffun_type, gfun_type)
            X0[:, i] = X_temp0
            X1[:, i] = X_temp1
    # pd.DataFrame(noise_matrix).to_csv("Repository/5nodes10edges/noise_matrix.csv", index=False, header=False)
    return dv_ls, admissible_vars, X, X_cf, X0, X1

class DAG_simulation:
    """Generate synthetic data.
    Key instance variables:
        X (numpy.ndarray): [n, d] data matrix.
        B (numpy.ndarray): [d, d] weighted adjacency matrix of DAG.
        B_bin (numpy.ndarray): [d, d] binary adjacency matrix of DAG.
    Code modified from:
        https://github.com/xunzheng/notears/blob/master/notears/utils.py
    """
    _logger = logging.getLogger(__name__)

    def __init__(self, d, s, lb=0.1, ub=2.0):
        """Initialize self.
        Args:
            d (int): Number of nodes.
            s (int): Number of edges.
            lb (float): Lower bound of weights
            ub (float): Upper bound of weights
            seed (int): Random seed. Default: 1.
        """
        self.d = d
        self.s = s
        self.B_ranges = ((-ub, -lb), (lb, ub))
        self._setup()

    def _setup(self):
        """Generate B_bin, B and X."""
        self.B_bin = DAG_simulation.simulate_dag(self.d, self.s)
        self.B = DAG_simulation.simulate_weight(self.B_bin, self.B_ranges)
        assert is_dag(self.B)
    
    @staticmethod
    def simulate_dag(d, s0):
        """Simulate random DAG with some expected number of edges.
        Args:
            d (int): num of nodes
            s0 (int): expected num of edges
        Returns:
            B (np.ndarray): [d, d] binary adj matrix of DAG
        """
        def _random_permutation(M):
            # np.random.permutation permutes first axis only
            P = np.random.permutation(np.eye(M.shape[0]))
            return P.T @ M @ P

        def _topologically_permutation(M):
            ordered_vertices = list(nx.topological_sort(nx.DiGraph(M)))
            # ordered_vertices = [9, 7, 6, 8, 2, 1, 5, 4, 3, 0]
            P = np.eye(M.shape[0])[:, ordered_vertices]
            return P.T @ M @ P

        def _random_acyclic_orientation(B_und):
            return np.tril(_random_permutation(B_und), k=-1)

        def _graph_to_adjmat(G):
            return np.array(G.get_adjacency().data)

        # Erdos-Renyi
        G_und = ig.Graph.Erdos_Renyi(n=d, m=s0)
        B_und = _graph_to_adjmat(G_und)
        B = _random_acyclic_orientation(B_und)

        # B_perm = _random_permutation(B)
        B_perm = _topologically_permutation(B)
        assert ig.Graph.Adjacency(B_perm.tolist()).is_dag()
        return B_perm

    @staticmethod
    def simulate_weight(B_bin, B_ranges):
        """Simulate the weights of B_bin.
        Args:
            B_bin (numpy.ndarray): [d, d] binary adjacency matrix of DAG.
            B_ranges (tuple): Disjoint weight ranges.
            rs (numpy.random.RandomState): Random number generator.
                Default: np.random.RandomState(1).
        Returns:
            numpy.ndarray: [d, d] weighted adjacency matrix of DAG.
        """
        B = np.zeros(B_bin.shape)
        S = np.random.randint(len(B_ranges), size=B.shape)  # Which range
        for i, (low, high) in enumerate(B_ranges):
            U = np.random.uniform(low=low, high=high, size=B.shape)
            B += B_bin * (S == i) * U
        return B

