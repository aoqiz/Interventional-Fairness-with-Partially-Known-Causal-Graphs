#%%
import os
import pickle
import argparse
import numpy as np
import pandas as pd
from scipy import stats
from fitter import Fitter
import torch
from torch.utils.data import DataLoader, Dataset
import mdnetwork as mdn
from DataLoader.utils import set_random_seed


class MyDataset(Dataset):
    def __init__(self, df, bucket, parent=None):
        if parent == None:
            self.parent = False
            self.x = df[bucket]
        else:
            self.parent = True
            self.x = df[parent]
            self.y = df[bucket]
        
    def __getitem__(self, idx):
        if self.parent == True:
            return self.x.iloc[idx].to_numpy(), self.y.iloc[idx].to_numpy()	
        else:
            return self.x.iloc[idx].to_numpy()

    def __len__(self):
        return len(self.x)
    

def main(args):
    # number of nodes
    d = args.num_of_nodes
    # number of edges
    s = args.num_of_edges
    # number of graphs
    k = args.num_of_graphs
    # num_of_admissible_vars
    adm = args.num_of_admissible_vars
    # number of the samples
    nobs = 1000

    print("{} nodes {} edges {} graphs {} admissible variables: ".format(d, s, k, adm))

    # ------path setting------
    dir = "Repository_adm={}/{}nodes{}edges".format(adm, d, s)
    observational_data_path = "{}/observational_data_{}.csv".format(dir, k)
    interventional0_data_truth_path = "{}/interventional0_data_truth_{}.csv".format(dir, k)
    interventional1_data_truth_path = "{}/interventional1_data_truth_{}.csv".format(dir, k)
    relation_path = '{}/interventional0_bp_{}.txt'.format(dir, k)

    # ------file read in and setting up------
    df = pd.read_csv(observational_data_path, delimiter=',',header=None)
    train_size, test_size = int(len(df) * 0.8), len(df) - int(len(df) * 0.8)

    pairs,inputs,protected,nonDes = [],[],0,[]
    with open(relation_path, 'r') as f:
        n,protected = [int(char)-1 for char in f.readline().strip().split(' ')]
        for _ in range(n+1):
            b,p = f.readline().split('\t')
            b = [int(char)-1 for char in b.strip().split(' ')[1:]]
            p = [int(char)-1 for char in p.strip().split(' ')[1:]]
            if not p:  nonDes.extend(b)
            else:  
                pairs.append([b,p])
                input = MyDataset(df, b, p)
                inputs.append(input)

    # training process
    epochs = 50
    lr = 1e-2
    batch_size = 64
    num_Gaussian = 1
    num_hidden = 16

    path = "{}/models".format(dir)
    if not os.path.exists(path):
        os.mkdir(path)

    for i in range(len(inputs)):
        wfile = "{}/models/graph_{}_model_{}".format(dir, k, i)

        model = mdn.MDN(len(pairs[i][1]), len(pairs[i][0]), num_Gaussian, num_hidden)
        model = model.double()

        trainData, testData = torch.utils.data.random_split(inputs[i], [train_size, test_size])
        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=1e-3)
        trainSet = DataLoader(dataset=trainData, shuffle=True, drop_last=False, batch_size=batch_size)
        testSet = DataLoader(dataset=testData, shuffle=False, drop_last=False, batch_size=len(testData))

        print("...........Start graph_{}_model_{}...........".format(k, i))
        for epoch in range(epochs):
            train_loss, train_count = 0, 0
            for x, y in trainSet:
                train_count += 1
                model.zero_grad()
                pi, sigma, mu = model(x)

                loss = mdn.mdn_loss(pi, sigma, mu, y)
                
                train_loss += loss.detach().numpy()
                loss.backward()
                optimizer.step()
                
            test_loss, test_count = 0, 0
            with torch.no_grad():
                for x, y in testSet:
                    test_count += 1
                    pi, sigma, mu = model(x)
                    test_loss += mdn.mdn_loss(pi, sigma, mu, y).detach().numpy()

            print('Epoch: {}, TrainLoss: {:.3f}, TestLoss: {:.3f}'.format(epoch + 1, train_loss / train_count,
                                                                        test_loss / test_count))
        file = open(wfile+'.mdl', 'wb')
        pickle.dump(model, file)
        file.close()

        print("...........graph_{}_model_{} done!...........".format( k, i))


    # fit the nonDesendent variables
    params = []
    for nonDe in nonDes:
        f = Fitter(df[nonDe], distributions=['norm'])
        f.fit()
        params.append(f.fitted_param['norm'])

    gened_data0 = pd.DataFrame(np.zeros((nobs,d-1)))
    gened_data1 = pd.DataFrame(np.zeros((nobs,d-1)))

    # generate non-Desendent variables
    for i in range(len(nonDes)):
        gened_data0[nonDes[i]] = stats.norm.rvs(*params[i],size=nobs)
        gened_data1[nonDes[i]] = stats.norm.rvs(*params[i],size=nobs)

    # generate intervention variables
    gened_data1[protected] = np.ones((nobs,1))

    # generate variable 3 and so on and so forth
    for i in range(len(inputs)):
        wfile = "{}/models/graph_{}_model_{}".format(dir, k, i)
        file = open(wfile+'.mdl', 'rb')
        model = pickle.load(file)
        file.close()
        
        input0 = MyDataset(gened_data0, pairs[i][1])
        input1 = MyDataset(gened_data1, pairs[i][1])
        dataSet0 = DataLoader(dataset=input0, shuffle=False, drop_last=False, batch_size=len(input0))
        dataSet1 = DataLoader(dataset=input1, shuffle=False, drop_last=False, batch_size=len(input1))

        with torch.no_grad():
            for x in dataSet0:
                alpha, mu, sigma = model(x)
                gened_data0[pairs[i][0]] = mdn.sample(alpha, mu, sigma).numpy()
            
            for x in dataSet1:
                alpha, mu, sigma = model(x)
                gened_data1[pairs[i][0]] = mdn.sample(alpha, mu, sigma).numpy()

    # save the generated data
    interventional0_data_gene_path = "{}/interventional0_data_gene_{}.csv".format(dir, k)
    interventional1_data_gene_path = "{}/interventional1_data_gene_{}.csv".format(dir, k)
    gened_data0.to_csv(interventional0_data_gene_path, header=False, index=False)
    gened_data1.to_csv(interventional1_data_gene_path, header=False, index=False)



if __name__ == "__main__":
    #seed
    set_random_seed(532)
    torch.manual_seed(532)

    # ------Parameters setting------
    parser = argparse.ArgumentParser()
    parser.add_argument('num_of_nodes', type=int, help='the number of nodes')
    parser.add_argument('num_of_edges', type=int, help='the number of edges')
    parser.add_argument('num_of_graphs', type=int, help='the number of graphs')
    parser.add_argument('num_of_admissible_vars', type=int, help='the number of admissible variables')
    args = parser.parse_args()

    main(args)


