import os.path
import time
import datetime
import argparse
import numpy as np
import pandas as pd

import torch
from torch.utils.tensorboard import SummaryWriter

from model import Model, train, evaluation
from utils import load_checkpoint, set_random_seed, get_variable_index, save_results


# main function
def main(args):
    set_random_seed(args.seed)
    seeds_to_split_data = np.random.randint(low=0, high=9999, size=10)

    print(f"----------seeds_to_split_data: {seeds_to_split_data}------------")

    args.dir = "data"
    observational_data_path = "{}/observation_data.csv".format(args.dir)
    interventional0_data_gene_path = "{}/interventional0_data_gene.csv".format(args.dir)
    interventional1_data_gene_path = "{}/interventional1_data_gene.csv".format(args.dir)
    interventional0_data_truth_path = "{}/interventional0_data_real.csv".format(args.dir)
    interventional1_data_truth_path = "{}/interventional1_data_real.csv".format(args.dir)

    # Import data
    datasets = {}
    datasets['observation'] = pd.read_csv(observational_data_path, header=0, delimiter=',').astype(np.float32)
    datasets['interventional0'] = pd.read_csv(interventional0_data_gene_path, header=0, delimiter=',').astype(np.float32)
    datasets['interventional1'] = pd.read_csv(interventional1_data_gene_path, header=0, delimiter=',').astype(np.float32)
    datasets['interventional0_truth'] = pd.read_csv(interventional0_data_truth_path, header=0, delimiter=',').astype(np.float32)
    datasets['interventional1_truth'] = pd.read_csv(interventional1_data_truth_path, header=0, delimiter=',').astype(np.float32)


    args.protected = 'Age'
    args.outcome = 'Loan_Status'
    args.NonDes = []
    args.Des = list(set(datasets['observation'].columns) - set([args.protected, args.outcome]) )

    nodes = datasets['interventional0'].columns
    index_full, index_unaware, index_IFair = get_variable_index(nodes, args)
    args.mode_var = {'Full': index_full, 'Unaware': index_unaware}

    # ------Data Splitting------
    n0_real = len(datasets['interventional0_truth'])
    n1_real = len(datasets['interventional1_truth'])
    intervene_n = len(datasets['interventional0'])

    for i, seed in enumerate(seeds_to_split_data):
        print(i, seed)
        set_random_seed(seed)
        test_index = [np.random.choice(np.arange(0, n0_real), size=int(0.1 * n0_real), replace=False),
                    np.random.choice(np.arange(0, n1_real), size=int(0.1 * n1_real), replace=False)]
        val_index = [np.random.choice(np.array(list(set(np.arange(0, n0_real)).difference(test_index[0]))),
                                    size=int(0.1 * n0_real),
                                    replace=False),
                    np.random.choice(np.array(list(set(np.arange(0, n1_real)).difference(test_index[1]))),
                                    size=int(0.1 * n1_real),
                                    replace=False)]
        training_index = [np.array(list(set(np.arange(0, n0_real))
                                        .difference(np.concatenate((test_index[0], val_index[0]), axis=0)))),
                        np.array(list(set(np.arange(0, n1_real))
                                        .difference(np.concatenate((test_index[1], val_index[1]), axis=0))))]

        intervene_test_index = np.nan
        intervene_val_index = np.random.choice(np.arange(0, intervene_n), size=int(0.2 * intervene_n), replace=False)
        intervene_training_index = np.array(list(set(np.arange(0, intervene_n)).difference(intervene_val_index)))

        args.process2index = {"Train": (training_index, intervene_training_index), "Validation": (val_index, intervene_val_index),
                        "Test": (test_index, intervene_test_index)}


        ####################################
        #### Train and predict Baselines
        ####################################
        lambda_ = 0.0
        RMSE = np.zeros([1, len(args.mode_var)])
        Unfairness = np.zeros([1, len(args.mode_var)])
        Accuracy = np.zeros([1, len(args.mode_var)])
        for i, mode in enumerate(args.mode_var.keys()):
            set_random_seed(seed)
            var_ind = args.mode_var[mode]
            if var_ind == []:
                RMSE[0, i] = np.nan
                Unfairness[0, i] = np.nan
                Accuracy[0, i] = np.nan
            RMSE_cv = []
            Unfairness_cv = []
            Accuracy_cv = []
            # Train
            model = Model(dim=len(var_ind))
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0)
            train(args, model, datasets, optimizer, num_iters=args.BASE_ITER, eval_every=1, lambda_=lambda_, mode=mode)
            # Predict
            best_model = Model(dim=len(var_ind))
            load_checkpoint(f'{args.dir}/Model_{mode}.pt', best_model)
            test_loss_result, y0_test_pred, y1_test_pred = evaluation(args, best_model, datasets, mode=mode, lambda_=lambda_)
            test_loss, test_RMSE, test_unfairness, test_accuracy = test_loss_result

            RMSE_cv.append(test_RMSE.item())
            Unfairness_cv.append(test_unfairness.item())
            Accuracy_cv.append(test_accuracy.item())
            print("Finished Training {} with lambda={}!".format(mode, lambda_))
            RMSE[0, i] = round(sum(RMSE_cv) / len(RMSE_cv), 3)
            Unfairness[0, i] = round(sum(Unfairness_cv) / len(Unfairness_cv), 3)
            Accuracy[0, i] = round(sum(Accuracy_cv) / len(Accuracy_cv), 3)

            # # save data
            # results_dir = "results"
            # os.makedirs(results_dir, exist_ok=True)
            # y_test_pred_dir = "y_pred"
            # os.makedirs("{}/{}".format(results_dir, y_test_pred_dir), exist_ok=True)
            # pd.DataFrame(y0_test_pred.view(1, -1).numpy()).to_csv("{}/{}/{}_y0.csv".format(results_dir, y_test_pred_dir, mode),
            #                                               mode='a', index=False,
            #                                               header=False)
            # pd.DataFrame(y1_test_pred.view(1, -1).numpy()).to_csv("{}/{}/{}_y1.csv".format(results_dir, y_test_pred_dir, mode),
            #                                               mode='a', index=False,
            #                                               header=False)

        print('RMSE:', RMSE)
        print('Unfairness: ', Unfairness)
        print('Accuracy:', Accuracy)
        # save data
        save_results(args, RMSE, Unfairness, Accuracy, name="")


        #################################
        #### Train the model e-IFair (with generated interventionals)
        #################################
        lambdas = [2, 4, 6, 10, 15, 150]
        mode = 'e-IFair'
        RMSEIF = np.zeros([1, len(lambdas)])
        UnfairnessIF = np.zeros([1, len(lambdas)])
        AccuracyIF = np.zeros([1, len(lambdas)])
        for i, lambda_ in enumerate(lambdas):
            set_random_seed(seed)
            RMSEIF_cv = []
            UnfairnessIF_cv = []
            AccuracyIF_cv = []
            model = Model(dim=len(index_full))
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0)
            train(args, model, datasets, optimizer, num_iters=args.GENE_ITER, eval_every=1, lambda_=lambda_, mode=mode, ground_truth=False)
            best_model = Model(dim=len(index_full))
            load_checkpoint(f'{args.dir}/Model_{mode}.pt', best_model)
            test_loss_result, y0_test_pred, y1_test_pred = evaluation(args, best_model, datasets, mode=mode, lambda_=lambda_,
                                                            ground_truth=False)
            test_loss, test_RMSE, test_unfairness, test_accuracy = test_loss_result

            RMSEIF_cv.append(test_RMSE.item())
            UnfairnessIF_cv.append(test_unfairness.item())
            AccuracyIF_cv.append(test_accuracy.item())
            print("Finished Training {} with lambda={}!".format(mode, lambda_))
            RMSEIF[0, i] = round(sum(RMSEIF_cv) / len(RMSEIF_cv), 3)
            UnfairnessIF[0, i] = round(sum(UnfairnessIF_cv) / len(UnfairnessIF_cv), 3)
            AccuracyIF[0, i] = round(sum(AccuracyIF_cv) / len(AccuracyIF_cv), 3)

            # pd.DataFrame(y0_test_pred.view(1, -1).numpy()).to_csv(
            #     "{}/{}/{}_{}_y0.csv".format(results_dir, y_test_pred_dir, mode, lambda_), mode='a',
            #     index=False,
            #     header=False)
            # pd.DataFrame(y1_test_pred.view(1, -1).numpy()).to_csv(
            #     "{}/{}/{}_{}_y1.csv".format(results_dir, y_test_pred_dir, mode, lambda_), mode='a',
            #     index=False,
            #     header=False)

        print('RMSEIF:', RMSEIF)
        print('UnfairnessIF: ', UnfairnessIF)
        print('AccuracyIF: ', AccuracyIF)
        # save data
        save_results(args, RMSEIF, UnfairnessIF, AccuracyIF, name="IF", lambdas=lambdas)


if __name__ == "__main__":
    # ------Parameters setting------
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=90, help='the seed we chosen')
    parser.add_argument('--BASE_ITER', type=int, default=1000, help='Iteration number when lambda is small')
    parser.add_argument('--GENE_ITER', type=int, default=1000, help='Iteration number when lambda is large')
    parser.add_argument('--device', type=str, default="cpu", help='the device we try to use for ...')
    
    args = parser.parse_args()

    main(args)



