#%%
import os.path
import time
import datetime
import argparse
import numpy as np
import pandas as pd

import torch
from torch.utils.tensorboard import SummaryWriter

from model import Model, train, evaluation
from utils import load_checkpoint, set_random_seed, data_spliting, save_results


# main function
def main(args):
    set_random_seed(args.seed)

    print("{} nodes {} edges {} graphs {} admissible variables: ".format(args.num_nodes,args.num_edges, args.graph_id, args.adm))

    # data file path
    args.dir = "Repository_adm={}/{}nodes{}edges".format(args.adm, args.num_nodes, args.num_edges)
    writer = SummaryWriter("{}/log_{}.log".format(args.dir, args.graph_id))
    observational_data_path = "{}/observational_data_{}.csv".format(args.dir, args.graph_id)
    interventional0_data_truth_path = "{}/interventional0_data_truth_{}.csv".format(args.dir, args.graph_id)
    interventional1_data_truth_path = "{}/interventional1_data_truth_{}.csv".format(args.dir, args.graph_id)
    interventional0_data_gene_path = "{}/interventional0_data_gene_{}.csv".format(args.dir, args.graph_id)
    interventional1_data_gene_path = "{}/interventional1_data_gene_{}.csv".format(args.dir, args.graph_id)
    config_path = "{}/config_{}.txt".format(args.dir, args.graph_id)
    relation_path = "{}/relation_{}.txt".format(args.dir, args.graph_id)

    # Import data
    datasets = {}
    datasets['observation'] = np.genfromtxt(observational_data_path, skip_header=0, delimiter=',').astype(np.float32)
    datasets['interventional0'] = np.genfromtxt(interventional0_data_gene_path, skip_header=0, delimiter=',').astype(np.float32)
    datasets['interventional1'] = np.genfromtxt(interventional1_data_gene_path, skip_header=0, delimiter=',').astype(np.float32)
    datasets['interventional0_truth'] = np.genfromtxt(interventional0_data_truth_path, skip_header=0, delimiter=',').astype(np.float32)
    datasets['interventional1_truth'] = np.genfromtxt(interventional1_data_truth_path, skip_header=0, delimiter=',').astype(np.float32)

    
    # Import ancestral relations
    Lines = open(relation_path, 'r').readlines()
    args.relation = {'defNonDes': [int(i) - 1 for i in Lines[0].strip().split()],
                'possDes': [int(i) - 1 for i in Lines[1].strip().split()],
                'defDes': [int(i) - 1 for i in Lines[2].strip().split()],
                'NonDes': [int(i) - 1 for i in Lines[3].strip().split()],
                'Des': [int(i) - 1 for i in Lines[4].strip().split()]}

    # Import config
    args.config = pd.read_csv(config_path, delimiter=',').T.to_dict()[0]
    
    # Data spliting
    args = data_spliting(datasets, args)


    ####################################
    #### Train and predict Baselines
    ####################################
    lambda_ = 0.0; num_iters = args.Iter
    RMSE = np.zeros([1, len(args.mode_var)])
    Unfairness = np.zeros([1, len(args.mode_var)])
    for i, mode in enumerate(args.mode_var.keys()):
        if mode == 'Full':
            num_iters = args.Iter2
        set_random_seed(args.seed)
        var_ind = args.mode_var[mode]
        if var_ind == []:
            RMSE[0, i] = np.nan
            Unfairness[0, i] = np.nan
        RMSE_cv = []
        Unfairness_cv = []
        # Train
        print("\n----------- Start Training {} with lambda={} -----------".format(mode, lambda_))
        model = Model(dim=len(var_ind))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0)
        train(args, model, datasets, optimizer, writer, num_iters=num_iters, lambda_=lambda_, mode=mode)
        # Predict
        best_model = Model(dim=len(var_ind))
        load_checkpoint(f'{args.dir}/Model_{mode}.pt', best_model)
        test_loss_result, y0_test_pred, y1_test_pred = evaluation(args, best_model, datasets, mode=mode, lambda_=lambda_)
        test_loss, test_RMSE, test_unfairness = test_loss_result
        RMSE_cv.append(test_RMSE.item())
        Unfairness_cv.append(test_unfairness.item())
        print("----------- Finished Training {} with lambda={}! -----------\n".format(mode, lambda_))
        RMSE[0, i] = round(sum(RMSE_cv) / len(RMSE_cv), 3)
        Unfairness[0, i] = round(sum(Unfairness_cv) / len(Unfairness_cv), 3)

        # save predicted y
        y_test_pred_dir = "y_pred"
        os.makedirs("{}/{}".format(args.dir, y_test_pred_dir), exist_ok=True)
        pd.DataFrame(y0_test_pred.view(1, -1).numpy()).to_csv("{}/{}/{}_y0.csv".format(args.dir, y_test_pred_dir, mode),
                                                    mode='a', index=False, header=False)
        pd.DataFrame(y1_test_pred.view(1, -1).numpy()).to_csv("{}/{}/{}_y1.csv".format(args.dir, y_test_pred_dir, mode),
                                                    mode='a', index=False, header=False)

    print(RMSE)
    print(Unfairness)

    ## save data
    save_results(args, RMSE, Unfairness, name="")


    ##############################
    # Train the model e-IFair (with ground_truth interventionals)
    ##############################
    lambdas = [0, 0.5, 5, 20, 60, 100]
    mode = 'e-IFair'
    RMSEIF_truth = np.zeros([1, len(lambdas)])
    UnfairnessIF_truth = np.zeros([1, len(lambdas)])
    for i, lambda_ in enumerate(lambdas):
        num_iters = args.Iter2 if lambda_ > 5 else args.Iter
        set_random_seed(args.seed)
        RMSEIF_cv = []
        UnfairnessIF_cv = []
        model = Model(dim=len(args.mode_var['Full']))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0)
        train(args, model, datasets, optimizer, writer, num_iters=num_iters, lambda_=lambda_, mode=mode)
        best_model = Model(dim=len(args.mode_var['Full']))
        load_checkpoint(f'{args.dir}/Model_{mode}.pt', best_model)
        test_loss_result, y0_test_pred, y1_test_pred = evaluation(args, best_model, datasets, mode=mode, lambda_=lambda_)
        test_loss, test_RMSE, test_unfairness = test_loss_result
        RMSEIF_cv.append(test_RMSE.item())
        UnfairnessIF_cv.append(test_unfairness.item())
        print("Finished Training {} with lambda={}!".format(mode, lambda_))
        RMSEIF_truth[0, i] = round(sum(RMSEIF_cv) / len(RMSEIF_cv), 3)
        UnfairnessIF_truth[0, i] = round(sum(UnfairnessIF_cv) / len(UnfairnessIF_cv), 3)

        # save predicted y
        y_test_pred_dir = "y_pred"
        os.makedirs("{}/{}".format(args.dir, y_test_pred_dir), exist_ok=True)
        
        pd.DataFrame(y0_test_pred.view(1, -1).numpy()).to_csv(
            "{}/{}/{}_{}_y0_real.csv".format(args.dir, y_test_pred_dir, mode, lambda_), mode='a', index=False, header=False)
        pd.DataFrame(y1_test_pred.view(1, -1).numpy()).to_csv(
            "{}/{}/{}_{}_y1_real.csv".format(args.dir, y_test_pred_dir, mode, lambda_), mode='a', index=False, header=False)

    print(RMSEIF_truth)
    print(UnfairnessIF_truth)
    # save data
    save_results(args, RMSEIF_truth, UnfairnessIF_truth, name="IF_truth", lambdas=lambdas)


    # ################################
    # ### Train the model e-IFair (with generated interventionals)
    # ################################
    lambdas = [0, 0.5, 5, 20, 60, 100]
    mode = 'e-IFair'
    RMSEIF = np.zeros([1, len(lambdas)])
    UnfairnessIF = np.zeros([1, len(lambdas)])
    for i, lambda_ in enumerate(lambdas):
        set_random_seed(args.seed)
        RMSEIF_cv = []
        UnfairnessIF_cv = []
        model = Model(dim=len(args.mode_var['Full']))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0)
        num_iters = args.Iter2 if lambda_ > 5 else args.Iter
        train(args, model, datasets, optimizer, writer, num_iters=num_iters, 
              lambda_=lambda_, mode=mode, ground_truth=False)
        best_model = Model(dim=len(args.mode_var['Full']))
        load_checkpoint(f'{args.dir}/Model_{mode}.pt', best_model)
        test_loss_result, y0_test_pred, y1_test_pred = evaluation(args, best_model, datasets, mode=mode, lambda_=lambda_,
                                                        ground_truth=False)
        test_loss, test_RMSE, test_unfairness = test_loss_result
        RMSEIF_cv.append(test_RMSE.item())
        UnfairnessIF_cv.append(test_unfairness.item())
        print("Finished Training {} with lambda={}!".format(mode, lambda_))
        RMSEIF[0, i] = round(sum(RMSEIF_cv) / len(RMSEIF_cv), 3)
        UnfairnessIF[0, i] = round(sum(UnfairnessIF_cv) / len(UnfairnessIF_cv), 3)

        # save predicted y
        y_test_pred_dir = "y_pred"
        os.makedirs("{}/{}".format(args.dir, y_test_pred_dir), exist_ok=True)
        
        pd.DataFrame(y0_test_pred.view(1, -1).numpy()).to_csv(
            "{}/{}/{}_{}_y0_gene.csv".format(args.dir, y_test_pred_dir, mode, lambda_), mode='a', index=False, header=False)
        pd.DataFrame(y1_test_pred.view(1, -1).numpy()).to_csv(
            "{}/{}/{}_{}_y1_gene.csv".format(args.dir, y_test_pred_dir, mode, lambda_), mode='a', index=False, header=False)
    
    print(RMSEIF)
    print(UnfairnessIF)
    # save data
    save_results(args, RMSEIF, UnfairnessIF, name="IF", lambdas=lambdas)

    print(time.time())
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))



if __name__ == "__main__":
    # ------Parameters setting------
    parser = argparse.ArgumentParser()
    parser.add_argument('num_nodes', type=int, help='the number of nodes')
    parser.add_argument('num_edges', type=int, help='the number of edges')
    parser.add_argument('graph_id', type=int, help='the number of graphs')
    parser.add_argument('adm', type=int, help='the number of admissible variables')
    parser.add_argument('--seed', type=int, default=532, help='the seed we chosen')
    parser.add_argument('--Iter', type=int, default=400, help='the seed we chosen')
    parser.add_argument('--Iter2', type=int, default=500, help='the seed we chosen')
    
    args = parser.parse_args()

    main(args)


