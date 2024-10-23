import os
import torch
import random
import numpy as np
import pandas as pd


def set_random_seed(seed):
    random.seed(seed)
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


def get_variable_index(args):
    args.num_nodes, args.protected, args.outcome, args.relation['NonDes']
    
    index_full = [i for i in list(range(0, args.num_nodes)) if i != args.outcome]
    index_unaware = [i for i in list(range(0, args.num_nodes)) if i != args.protected and i != args.outcome]
    index_IFair = list(set(args.relation['NonDes']).difference([args.outcome]))
    return index_full, index_unaware, index_IFair


def data_spliting(datasets, args):
    num_observe = len(datasets['observation'])
    num_intervene = len(datasets['interventional0_truth'])
    args.protected = args.config['protected'] - 1
    args.outcome = args.config['outcome'] - 1

    test_index = np.random.choice(np.arange(0, num_observe), size=int(0.1 * num_observe), replace=False)
    val_index = np.random.choice(np.array(list(set(np.arange(0, num_observe)).difference(test_index))), 
                                 size=int(0.1 * num_observe), replace=False)
    training_index = np.array(list(set(np.arange(0, num_observe)).difference(np.concatenate((test_index, 
                                                                                val_index), axis=0))))

    intervene_test_index = np.arange(0, num_intervene)[num_intervene//2:]
    intervene_val_index = np.random.choice(np.arange(0, num_intervene//2), size=int(0.2 * num_intervene//2), 
                                           replace=False)
    intervene_training_index = np.array(list(set(np.arange(0, num_intervene//2)).difference(intervene_val_index)))

    index_full, index_unaware, index_IFair = get_variable_index(args)
    args.mode_var = {'Full': index_full, 'Unaware': index_unaware, 'Fair': index_IFair}
    args.process2index = {"Train": (training_index, intervene_training_index), "Validation": (val_index, intervene_val_index),
                    "Test": (test_index, intervene_test_index)}
    
    return args


def data_preprocess(datasets, process2index, ground_truth=True, process="Train"):
    index, intervene_index = process2index[process]
    data = datasets['observation'][index, ]
    # To extract the interventional dataset for ground_truth e-IFair model; extract the interventional test set.
    if ground_truth == True or process == "Test":
        interventional0_data = datasets['interventional0_truth'][intervene_index, ]
        interventional1_data = datasets['interventional1_truth'][intervene_index, ]
    else:
        interventional0_data = datasets['interventional0'][intervene_index, ]
        interventional1_data = datasets['interventional1'][intervene_index, ]
    return data, interventional0_data, interventional1_data


def save_results(args, RMSE, Unfairness, name="", lambdas=[]):
    if (not os.path.exists("{}/RMSE{}.csv".format(args.dir, name))) or (not os.path.exists("{}/Unfairness{}.csv".format(args.dir, name))):
        if name == "":
            pd_header = pd.DataFrame(columns=args.mode_var.keys())
        elif name.startswith("IF"):
            pd_header = pd.DataFrame(columns=[str(i) + "IF" for i in lambdas])
        pd.DataFrame(pd_header).to_csv("{}/RMSE{}.csv".format(args.dir, name), index=False, header=True)
        pd.DataFrame(pd_header).to_csv("{}/Unfairness{}.csv".format(args.dir, name), index=False, header=True)
    pd.DataFrame(RMSE).to_csv("{}/RMSE{}.csv".format(args.dir, name), mode='a', index=False, header=False)
    pd.DataFrame(Unfairness).to_csv("{}/Unfairness{}.csv".format(args.dir, name), mode='a', index=False, header=False)

    print("saved successfully!")
