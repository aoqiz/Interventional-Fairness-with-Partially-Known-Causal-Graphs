import torch
import torch.nn as nn
import torch.nn.functional as F

import mmd
from utils import data_preprocess, save_checkpoint


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


def loss_func(y_pred, y0_pred, y1_pred, y, lambda_):
    '''loss = prediction loss + lambda * unfairness'''
    prediction_loss = F.mse_loss(y_pred, y) 

    if lambda_ == 0:
        return prediction_loss, torch.abs(torch.mean(y0_pred) - torch.mean(y0_pred)), prediction_loss

    unfairness = mmd.mix_rbf_mmd2(y0_pred, y1_pred)
    loss = prediction_loss + lambda_ * unfairness
    return prediction_loss, unfairness, loss


# training and evaluation
def train(args, model, datasets, optimizer, num_iters=2000, eval_every=10, 
          lambda_=0.0, mode="Full", ground_truth=True):
    training_data, training_interventional0_data, training_interventional1_data = data_preprocess(datasets, 
                                                            args.process2index, process="Train")
    val_data, val_interventional0_data, val_interventional1_data = data_preprocess(datasets, args.process2index, 
                                                            process="Validation")

    model.to(args.device)

    if mode == "Full" or mode == "e-IFair":
        x = torch.from_numpy(training_data[args.mode_var['Full']].to_numpy())
        x0 = torch.from_numpy(training_interventional0_data[args.mode_var['Full']].to_numpy())
        x1 = torch.from_numpy(training_interventional1_data[args.mode_var['Full']].to_numpy())
        x_val = torch.from_numpy(val_data[args.mode_var['Full']].to_numpy())
        x0_val = torch.from_numpy(val_interventional0_data[args.mode_var['Full']].to_numpy())
        x1_val = torch.from_numpy(val_interventional1_data[args.mode_var['Full']].to_numpy())
    if mode == "Unaware":
        x = torch.from_numpy(training_data[args.mode_var['Unaware']].to_numpy())
        x0 = torch.from_numpy(training_interventional0_data[args.mode_var['Unaware']].to_numpy())
        x1 = torch.from_numpy(training_interventional1_data[args.mode_var['Unaware']].to_numpy())
        x_val = torch.from_numpy(val_data[args.mode_var['Unaware']].to_numpy())
        x0_val = torch.from_numpy(val_interventional0_data[args.mode_var['Unaware']].to_numpy())
        x1_val = torch.from_numpy(val_interventional1_data[args.mode_var['Unaware']].to_numpy())
    if mode == "Fair":
        x = torch.from_numpy(training_data[args.mode_var['Fair']].to_numpy())
        x0 = torch.from_numpy(training_interventional0_data[args.mode_var['Fair']].to_numpy())
        x1 = torch.from_numpy(training_interventional1_data[args.mode_var['Fair']].to_numpy())
        x_val = torch.from_numpy(val_data[args.mode_var['Fair']].to_numpy())
        x0_val = torch.from_numpy(val_interventional0_data[args.mode_var['Fair']].to_numpy())
        x1_val = torch.from_numpy(val_interventional1_data[args.mode_var['Fair']].to_numpy())
    y = torch.unsqueeze(torch.from_numpy(training_data[args.outcome].to_numpy()), 1)
    y_val = torch.unsqueeze(torch.from_numpy(val_data[args.outcome].to_numpy()), 1)

    # training loop
    best_val_loss = float("Inf")
    model.train()

    x, x0, x1 = x.to(args.device), x0.to(args.device), x1.to(args.device)
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
            x_val, x0_val, x1_val = x_val.to(args.device), x0_val.to(args.device), x1_val.to(args.device)
            with torch.no_grad():
                y_val_pred = model(x_val)
                y0_val_pred = model(x0_val)
                y1_val_pred = model(x1_val)
                val_prediction_loss, val_unfairness, val_loss = loss_func(y_val_pred, y0_val_pred, y1_val_pred, y_val,
                                                                          lambda_=lambda_)

                # Record training loss from each iter into the writer
                args.writer.add_scalar(mode + str(lambda_) + '/' + str(ground_truth) + '/Train/Loss', train_loss.item(), i)
                args.writer.add_scalar(mode + str(lambda_) + '/' + str(ground_truth) + '/Train/PredictionLoss', train_prediction_loss.item(), i)
                args.writer.add_scalar(mode + str(lambda_) + '/' + str(ground_truth) + '/Train/Unfairness', train_unfairness.item(), i)
                args.writer.flush()
                # Record validation loss from each iter into the writer
                args.writer.add_scalar(mode + str(lambda_) + '/' + str(ground_truth) + '/Validation/Loss', val_loss.item(), i)
                args.writer.add_scalar(mode + str(lambda_) + '/' + str(ground_truth) + '/Validation/PredictionLoss', val_prediction_loss.item(), i)
                args.writer.add_scalar(mode + str(lambda_) + '/' + str(ground_truth) + '/Validation/Unfairness', val_unfairness.item(), i)
                args.writer.flush()

                # checkpoint
                if best_val_loss > val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(f'{args.dir}/Model_{mode}.pt', model, best_val_loss)
    return


def evaluation(args, model, datasets, mode="Full", lambda_=0.0, ground_truth=True):
    test_data, test_interventional0_data, test_interventional1_data = data_preprocess(datasets, args.process2index,
                                                                                       process="Test")
    if mode == "Full" or mode == "e-IFair":
        x_test = torch.from_numpy(test_data[args.mode_var['Full']].to_numpy())
        x0_test = torch.from_numpy(test_interventional0_data[args.mode_var['Full']].to_numpy())
        x1_test = torch.from_numpy(test_interventional1_data[args.mode_var['Full']].to_numpy())
    if mode == "Unaware":
        x_test = torch.from_numpy(test_data[args.mode_var['Unaware']].to_numpy())
        x0_test = torch.from_numpy(test_interventional0_data[args.mode_var['Unaware']].to_numpy())
        x1_test = torch.from_numpy(test_interventional1_data[args.mode_var['Unaware']].to_numpy())
    if mode == "Fair":
        x_test = torch.from_numpy(test_data[args.mode_var['Fair']].to_numpy())
        x0_test = torch.from_numpy(test_interventional0_data[args.mode_var['Fair']].to_numpy())
        x1_test = torch.from_numpy(test_interventional1_data[args.mode_var['Fair']].to_numpy())
    y_test = torch.unsqueeze(torch.from_numpy(test_data[args.outcome].values), 1)
    if mode == "Full" or mode == "Unaware" or mode == "Fair" or (mode == 'e-IFair' and lambda_==0):
        lambda_ = 1e-16
        assert(lambda_ != 0)

    model.eval()
    x_test, x0_test, x1_test, y_test = x_test.to(args.device), x0_test.to(args.device), x1_test.to(args.device), y_test.to(args.device)
    with torch.no_grad():
        y_test_pred = model(x_test)
        y0_test_pred = model(x0_test)
        y1_test_pred = model(x1_test)
        test_prediction_loss, test_unfairness, test_loss = loss_func(y_test_pred, y0_test_pred, y1_test_pred, y_test,
                                                                     lambda_=lambda_)
        
        test_loss_result = (test_loss, torch.sqrt(test_prediction_loss), test_unfairness)
    return test_loss_result, y0_test_pred, y1_test_pred
