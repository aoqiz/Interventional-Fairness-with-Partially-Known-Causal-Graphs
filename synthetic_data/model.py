import torch
import torch.nn as nn
import torch.nn.functional as F

import mmd
from utils import data_preprocess, save_checkpoint


# model definition
class Model(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.net = nn.Sequential(
            nn.Linear(self.dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        y = self.net(x)
        return y


# utils functions
def loss_func(y_pred, y0_pred, y1_pred, y, lambda_):
    prediction_loss = F.mse_loss(y_pred, y)

    if lambda_ == 0:
        return prediction_loss, 0, prediction_loss

    unfairness = mmd.mix_rbf_mmd2(y0_pred, y1_pred)
    loss = prediction_loss + lambda_ * unfairness
    return prediction_loss, unfairness, loss


# training and evaluation
def train(args, model, datasets, optimizer, writer, num_iters=2000, eval_every=10, 
          lambda_=0.0, mode="Full", ground_truth=True):
    training_data, training_interventional0_data, training_interventional1_data = data_preprocess(datasets, 
                                                            args.process2index, ground_truth, process="Train")
    val_data, val_interventional0_data, val_interventional1_data = data_preprocess(datasets, args.process2index, 
                                                            ground_truth, process="Validation")

    if mode == "Full" or mode == "e-IFair":
        x = torch.from_numpy(training_data[:, args.mode_var['Full']])
        x0 = torch.from_numpy(training_interventional0_data[:, args.mode_var['Full']])
        x1 = torch.from_numpy(training_interventional1_data[:, args.mode_var['Full']])
        x_val = torch.from_numpy(val_data[:, args.mode_var['Full']])
        x0_val = torch.from_numpy(val_interventional0_data[:, args.mode_var['Full']])
        x1_val = torch.from_numpy(val_interventional1_data[:, args.mode_var['Full']])
    if mode == "Unaware":
        x = torch.from_numpy(training_data[:, args.mode_var['Unaware']])
        x0 = torch.from_numpy(training_interventional0_data[:, args.mode_var['Unaware']])
        x1 = torch.from_numpy(training_interventional1_data[:, args.mode_var['Unaware']])
        x_val = torch.from_numpy(val_data[:, args.mode_var['Unaware']])
        x0_val = torch.from_numpy(val_interventional0_data[:, args.mode_var['Unaware']])
        x1_val = torch.from_numpy(val_interventional1_data[:, args.mode_var['Unaware']])
    if mode == "Fair":
        x = torch.from_numpy(training_data[:, args.mode_var['Fair']])
        x0 = torch.from_numpy(training_interventional0_data[:, args.mode_var['Fair']])
        x1 = torch.from_numpy(training_interventional1_data[:, args.mode_var['Fair']])
        x_val = torch.from_numpy(val_data[:, args.mode_var['Fair']])
        x0_val = torch.from_numpy(val_interventional0_data[:, args.mode_var['Fair']])
        x1_val = torch.from_numpy(val_interventional1_data[:, args.mode_var['Fair']])
    y = torch.unsqueeze(torch.from_numpy(training_data[:, args.outcome]), 1)
    y_val = torch.unsqueeze(torch.from_numpy(val_data[:, args.outcome]), 1)

    # training loop
    best_val_loss = float("Inf")
    model.train()
    for i in range(num_iters):
        y_pred = model(x)
        y0_pred = model(x0)
        y1_pred = model(x1)
        train_prediction_loss, train_unfairness, train_loss = loss_func(y_pred, y0_pred, y1_pred, y, lambda_=lambda_)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        # evaluation step
        if i % eval_every == 0:
            model.eval()
            with torch.no_grad():
                y_val_pred = model(x_val)
                y0_val_pred = model(x0_val)
                y1_val_pred = model(x1_val)
                val_prediction_loss, val_unfairness, val_loss = loss_func(y_val_pred, y0_val_pred, y1_val_pred, y_val,
                                                                          lambda_=lambda_)

                # Record training loss from each iter into the writer
                writer.add_scalar(mode + str(lambda_) + '/' + str(ground_truth) + '/Train/Loss', train_loss.item(), i)
                writer.add_scalar(mode + str(lambda_) + '/' + str(ground_truth) + '/Train/PredictionLoss', train_prediction_loss.item(), i)
                writer.add_scalar(mode + str(lambda_) + '/' + str(ground_truth) + '/Train/Unfairness', train_unfairness.item() if lambda_!=0 else 0, i)
                writer.flush()
                # Record validation loss from each iter into the writer
                writer.add_scalar(mode + str(lambda_) + '/' + str(ground_truth) + '/Validation/Loss', val_loss.item(), i)
                writer.add_scalar(mode + str(lambda_) + '/' + str(ground_truth) + '/Validation/PredictionLoss', val_prediction_loss.item(), i)
                writer.add_scalar(mode + str(lambda_) + '/' + str(ground_truth) + '/Validation/Unfairness', val_unfairness.item() if lambda_!=0 else 0, i)
                writer.flush()

                # print progress
                print(f"iter: {i}, trian ttloss: {round(train_loss.item(),2)}, rmse={round(train_prediction_loss.item(), 2)}, unfair={round(train_unfairness.item() if lambda_!=0 else 0,2)}")
                print(f"------validation ttloss: {round(val_loss.item(),2)}, rmse={round(val_prediction_loss.item(), 2)}, unfair={round(val_unfairness.item() if lambda_!=0 else 0,2)}")
                # checkpoint
                if best_val_loss > val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(f'{args.dir}/Model_{mode}.pt', model, best_val_loss)
    return


def evaluation(args, model, datasets, mode="Full", lambda_=0.0, ground_truth=True):
    test_data, test_interventional0_data, test_interventional1_data = data_preprocess(datasets, args.process2index,
                                                                                       ground_truth, process="Test")
    # test_data, test_interventional0_data, test_interventional1_data = data_preprocess(datasets, process2index, 
                                                                                    #  ground_truth, process="Train")
    if mode == "Full" or mode == "e-IFair":
        x_test = torch.from_numpy(test_data[:, args.mode_var['Full']])
        x0_test = torch.from_numpy(test_interventional0_data[:, args.mode_var['Full']])
        x1_test = torch.from_numpy(test_interventional1_data[:, args.mode_var['Full']])
    if mode == "Unaware":
        x_test = torch.from_numpy(test_data[:, args.mode_var['Unaware']])
        x0_test = torch.from_numpy(test_interventional0_data[:, args.mode_var['Unaware']])
        x1_test = torch.from_numpy(test_interventional1_data[:, args.mode_var['Unaware']])
    if mode == "Fair":
        x_test = torch.from_numpy(test_data[:, args.mode_var['Fair']])
        x0_test = torch.from_numpy(test_interventional0_data[:, args.mode_var['Fair']])
        x1_test = torch.from_numpy(test_interventional1_data[:, args.mode_var['Fair']])
    y_test = torch.unsqueeze(torch.from_numpy(test_data[:, args.outcome]), 1)
    if mode == "Full" or mode == "Unaware" or mode == "Fair" or (mode == 'e-IFair' and lambda_==0):
        lambda_ = 1e-16
        assert(lambda_ != 0)

    model.eval()
    with torch.no_grad():
        y_test_pred = model(x_test)
        y0_test_pred = model(x0_test)
        y1_test_pred = model(x1_test)
        test_prediction_loss, test_unfairness, test_loss = loss_func(y_test_pred, y0_test_pred, y1_test_pred, y_test,
                                                                     lambda_=lambda_)
        test_prediction_loss = torch.sqrt(test_prediction_loss)
        test_loss_result = (test_loss, test_prediction_loss, test_unfairness)
    return test_loss_result, y0_test_pred, y1_test_pred
