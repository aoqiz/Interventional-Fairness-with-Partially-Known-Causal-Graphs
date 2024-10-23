import os
import torch
# import random
import numpy as np
import pandas as pd


# Set Random Seed
def set_random_seed(seed):
    # random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Save and Load Functions
def save_checkpoint(save_path, model, valid_loss):
    if save_path == None:
        return
    state_dict = {'model_state_dict': model.state_dict(),
                  'valid_loss': valid_loss}
    torch.save(state_dict, save_path)
    # print(f'Model saved to ==> {save_path}')

def load_checkpoint(load_path, model):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if load_path == None:
        return
    state_dict = torch.load(load_path, map_location=device)
    # print(f'Model loaded from <== {load_path}')
    model.load_state_dict(state_dict['model_state_dict'])
    return


def get_variable_index(nodes, args):
    '''protected, outcome, NonDes are stings
       nodes: [node1, node2, node3, ...]
    '''
    index_full = [i for i in nodes if i != args.outcome]
    index_unaware = [i for i in nodes if i != args.protected and i != args.outcome]
    index_IFair = args.NonDes
    return index_full, index_unaware, index_IFair


def data_preprocess(datasets, process2index, process="Train"):
    index, intervene_index = process2index[process]

    real0 = datasets['interventional0_truth'].iloc[index[0], ]
    real1 = datasets['interventional1_truth'].iloc[index[1], ]
    data = pd.concat([real0, real1], axis=0)

    data = data.reset_index(drop=True)

    # To extract the interventional dataset for ground_truth e-IFair model; extract the interventional test set.
    if process == "Test":
        interventional0_data = real0
        interventional1_data = real1
    else:
        interventional0_data = datasets['interventional0'].iloc[intervene_index, ]
        interventional1_data = datasets['interventional1'].iloc[intervene_index, ]
    return (data, interventional0_data, interventional1_data)


def save_results(args, RMSE, Unfairness, name="", lambdas=[]):
    results_dir = "results"
    if (not os.path.exists("{}/RMSE{}.csv".format(results_dir, name))) or (not os.path.exists("{}/Unfairness{}.csv".format(results_dir, name))):
        if name == "":
            pd_header = pd.DataFrame(columns=args.mode_var.keys())
        elif name == "IF":
            pd_header = pd.DataFrame(columns=[str(i) + "IF" for i in lambdas])
        pd.DataFrame(pd_header).to_csv("{}/RMSE{}.csv".format(results_dir, name), index=False, header=True)
        pd.DataFrame(pd_header).to_csv("{}/Unfairness{}.csv".format(results_dir, name), index=False, header=True)
    pd.DataFrame(RMSE).to_csv("{}/RMSE{}.csv".format(results_dir, name), mode='a', index=False, header=False)
    pd.DataFrame(Unfairness).to_csv("{}/Unfairness{}.csv".format(results_dir, name), mode='a', index=False, header=False)
