from DataLoader.synthetic_dataset import DAG_simulation, simulate_data
from DataLoader.utils import show_dag, set_random_seed
import pandas as pd
import numpy as np
import os
import argparse
import networkx as nx


if __name__ == "__main__":
    # ------Parameters setting------
    parser = argparse.ArgumentParser()
    parser.add_argument('num_of_nodes', type=int, help='the number of nodes')
    parser.add_argument('num_of_edges', type=int, help='the number of edges')
    parser.add_argument('num_of_graphs', type=int, help='the number of graphs')
    parser.add_argument('num_of_admissible_vars', type=int, help='the number of admissible variables')
    args = parser.parse_args()
    #number of nodes
    d = args.num_of_nodes
    #number of edges
    s = args.num_of_edges
    # number of graphs
    k = args.num_of_graphs
    # num_of_admissible_vars
    adm = args.num_of_admissible_vars
    print("{} nodes {} edges {} graphs {} admissible variables: ".format(d, s, k, adm))

    # number of graphs
    G = 1
    # number of observations
    n = 5000
    # number of interventional data
    nn = 2*n
    #discrete variable and classes, for examples dv_ls=[4,5] dv_c_ls=[2,3] means node 4 is of [0,1] and node 5 of [0,1,2]
    dv_c_ls = [2]

    # save target path
    save_path = "Repository_adm={}/{}nodes{}edges".format(adm, d, s)
    os.makedirs(save_path, exist_ok=True)

    #seed
    set_random_seed(532 * k)

    DAG = DAG_simulation(d, s, lb=0.1, ub=1.0)
    B_bin = DAG.B_bin
    B = DAG.B
    ordered_vertices = list(nx.topological_sort(nx.DiGraph(B)))
    dv_ls, admissible_vars, observational_data, counterfactual_data, interventional0_data_truth, interventional1_data_truth = simulate_data(dv_c_ls, B, n, nn, adm, noise_type=0)
    outcome = ordered_vertices[-1]

    admissible_vals = [np.mean(observational_data[:, i]) for i in admissible_vars]
    admissible_vars_vals = {'var': [var + 1 for var in admissible_vars], 'val': admissible_vals}

    #save data
    pd.DataFrame(B_bin).to_csv("{}/adjacency_matrix_{}.csv".format(save_path, k), index=False, header=False)
    pd.DataFrame(B).to_csv("{}/weight_matrix_{}.csv".format(save_path, k), index=False, header=False)
    pd.DataFrame(observational_data).to_csv("{}/observational_data_{}.csv".format(save_path, k), index=False, header=False)
    pd.DataFrame(counterfactual_data).to_csv("{}/counterfactual_data_{}.csv".format(save_path, k), index=False, header=False)
    pd.DataFrame(interventional0_data_truth).to_csv("{}/interventional0_data_truth_{}.csv".format(save_path, k), index=False, header=False)
    pd.DataFrame(interventional1_data_truth).to_csv("{}/interventional1_data_truth_{}.csv".format(save_path, k), index=False, header=False)
    with open("{}/config_{}.txt".format(save_path, k), "w") as f:
        f.writelines('protected,protected_classes,outcome,sample_size,interventional_size\n'+ str(dv_ls[0]+1) + ','
                    + str(dv_c_ls[0]) + ',' + str(outcome+1) + ',' + str(n) + ',' + str(nn) + '\n')
    pd.DataFrame(admissible_vars_vals).to_csv("{}/admissible_{}.csv".format(save_path, k), index=False, header=True)
