#%%%
import pandas as pd

def get_dummy(df, features):
    df = pd.get_dummies(df, columns=features, drop_first=True) 
    return df

# ------path setting------
path = "PATH_TO_DATA"
dir = path+"/data"
data_path = "{}/student-mat.csv".format(dir)

# ------file read in and setting up------
data = pd.read_csv(data_path, delimiter=';',header=0)

data = data[data['absences'] <= 35]

data['Grade'] = (data['G3'] + data['G2'] + data['G1']) / 3
data.drop(['G1','G2','G3'], axis=1, inplace=True)


#%%
# ------fit values with parents------
import pickle
from scipy.stats import chi2_contingency


relation_path = '{}/bp_save_new.txt'.format(dir)

pairs,protected,inputs, nonDes, last_input = [],'',[],[], None
with open(relation_path, 'r') as f:
    n, protected = [word for word in f.readline().strip().split(' ')]

    for i in range(int(n)):
        b,p = f.readline().split('\t')
        b = [word for word in b.strip().split(' ')[1:]]
        p = [word for word in p.strip().split(' ')[1:]]

        if not p:  nonDes.append(b)
        elif i < 23:  
            pairs.append([b, p])
            wfile = "{}/models/model_{}".format(dir, i)
            print("...........Start model_{}...........".format(i))

            bucket = [data[a] for a in b]
            parent = [data[a] for a in p]

            ctgt = pd.crosstab(bucket, parent, margins=True)

            assert(chi2_contingency(ctgt.iloc[:-1, :-1])[1] < 0.01)

            cur_model = dict()
            attris = ctgt.columns.to_list()[:-1]
            for attri in attris:
                list_value = ctgt.index.to_list()[:-1]

                cur_nums = ctgt[attri].to_numpy()
                list_prob = cur_nums[:-1] / cur_nums[-1]
                cur_model[attri] = (list_value, list_prob)


            file = open(wfile+'.mdl', 'wb')
            pickle.dump(cur_model, file)
            file.close()
            print("...........model_{} done!...........".format(i))

        else:
            last_input = (b, p)


#%%
# fitting values with no parents
nonDes_params = []
for nonDe in nonDes:
    nonDes_params.append((data[nonDe].value_counts().keys().to_list(),  # params[i] = (list_values, list_prob)
                    data[nonDe].value_counts().to_list()))

#%%
# generate data
import pickle
import random
import numpy as np


RANDOM_SEED = 42    # 532
torch.manual_seed(RANDOM_SEED) 

n_sample = len(data)
nob0, nob1, d = n_sample, n_sample, len(data.columns)
gened_data0 = pd.DataFrame(np.zeros((nob0,d)))
gened_data1 = pd.DataFrame(np.zeros((nob1,d)))

gened_data0.columns = data.columns
gened_data1.columns = data.columns

# generate intervention variables
gened_data0[protected] = 'F'
gened_data1[protected] = 'M'

# generate data with no parents
for i in range(len(nonDes)):
    list_values, list_prob = nonDes_params[i]
    gened_data0[nonDes[i]] = random.choices(list_values, weights=list_prob, k=nob0)
    gened_data1[nonDes[i]] = random.choices(list_values, weights=list_prob, k=nob1)


#%%
assert(gened_data0.isnull().values.any() == False)

# generate conditional probability
for i in range(len(pairs)):
    outputs, inputs= pairs[i]

    input0 = gened_data0[inputs]
    input1 = gened_data1[inputs]

    wfile = "{}/models/model_{}".format(dir, i+14)
    file = open(wfile+'.mdl', 'rb')
    model = pickle.load(file)
    file.close()

    for j in range(nob0):
        gened_data0.loc[j,outputs] = input0.iloc[j].apply(
                lambda attri: random.choices(model[attri][0], model[attri][1]))[0][0]
        gened_data1.loc[j,outputs] = input1.iloc[j].apply(
                lambda attri: random.choices(model[attri][0], model[attri][1]))[0][0]

    print("...........model_{} used!...........".format(i+14))


#%%
# fitting last model
data_origin = data.copy()

import torch
from torch.utils.data import DataLoader
import numpy as np
import ctgnetwork as mdn
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(data.absences.values.reshape(-1,1))

data['absences'] = scaler.transform(data.absences.values.reshape(-1,1))

data = get_dummy(data, last_input[1])
p = [name for name in data.columns if any(map(lambda x: name.startswith(x), last_input[1]))]

#%%
RANDOM_SEED = 42    # 532   # 42
EPOCHS = 1000   # 200
LR = 1e-2
BATCH_SIZE = None
NUM_GAUSSIAN = 4  
NUM_HIDDEN = 16 
STEP_SIZE = 100  # 10
GAMMA = 0.3

torch.manual_seed(RANDOM_SEED) 
input =  mdn.MyDataset(data, b, p)

train_size, test_size = int(len(data) * 0.8), len(data) - int(len(data) * 0.8)

trainData, tempData = torch.utils.data.random_split(input, [train_size, test_size])
valData, testData = torch.utils.data.random_split(tempData, [test_size//2, test_size - test_size//2])
# trainSet = DataLoader(dataset=trainData, shuffle=True, drop_last=False, batch_size=BATCH_SIZE)
trainSet = DataLoader(dataset=trainData, shuffle=True, drop_last=False, batch_size=len(trainData))
valSet = DataLoader(dataset=valData, shuffle=False, drop_last=False, batch_size=len(valData))
testSet = DataLoader(dataset=testData, shuffle=False, drop_last=False, batch_size=len(testData))

#%%
from utils import save_checkpoint, load_checkpoint

wfile = "{}/models/model_{}".format(dir, 'final')

model = mdn.MDN(len(p), len(b), NUM_GAUSSIAN, NUM_HIDDEN)
loss_fn = mdn.mdn_loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=LR, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

print("...........Start model_{}...........".format('final'))
best_val_loss = float("Inf")

for epoch in range(EPOCHS):
    train_loss, train_count = 0, 0
    for x, y in trainSet:
        x, y = x.to(device), y.to(device)
        train_count += 1
        model.zero_grad()
        output = model(x)
        loss = loss_fn(output, y)
        train_loss += loss.detach().numpy()
        loss.backward()
        optimizer.step()

        val_loss, val_count = 0, 0
        with torch.no_grad():
            for x, y in valSet:
                x, y = x.to(device), y.to(device)
                val_count += 1
                output = model(x)
                val_loss += loss_fn(output, y).detach().numpy()
            
            cur_val_loss = val_loss / val_count
            if best_val_loss > cur_val_loss:
                best_val_loss = cur_val_loss
                save_checkpoint(f'{dir}/Model_final.pt', model, best_val_loss)

        print('Epoch: {}, TrainLoss: {:.3f}, ValLoss: {:.3f}'.format(epoch + 1, train_loss / train_count,
                                                                val_loss / val_count))
    scheduler.step()

    # val_loss, val_count = 0, 0
    # if i % 1 == 0:     
        # with torch.no_grad():
        #     for x, y in valSet:
        #         x, y = x.to(device), y.to(device)
        #         val_count += 1
        #         output = model(x)
        #         val_loss += loss_fn(output, y).detach().numpy()
            
        #     cur_val_loss = val_loss / val_count
        #     if best_val_loss > cur_val_loss:
        #         best_val_loss = cur_val_loss
        #         save_checkpoint(f'{dir}/Model_final.pt', model, best_val_loss)

        # print('Epoch: {}, TrainLoss: {:.3f}, ValLoss: {:.3f}'.format(epoch + 1, train_loss / train_count,
        #                                                         val_loss / val_count))


best_model = mdn.MDN(len(p), len(b), NUM_GAUSSIAN, NUM_HIDDEN)
load_checkpoint(f'{dir}/Model_final.pt', best_model)

file = open(wfile+'.mdl', 'wb')
pickle.dump(best_model, file)
file.close()
print("...........model_{} done!...........".format('final'))


#%%
# generate the last column
copy0 = gened_data0.copy()
copy1 = gened_data1.copy()

copy0 = get_dummy(copy0, last_input[1])
copy1 = get_dummy(copy1, last_input[1])

#%%
wfile = "{}/models/model_{}".format(dir, 'final')
file = open(wfile+'.mdl', 'rb')
model = pickle.load(file)
file.close()

input0 = mdn.MyDataset(copy0, p)
input1 = mdn.MyDataset(copy1, p)
dataSet0 = DataLoader(dataset=input0, shuffle=False, drop_last=False, batch_size=len(input0))
dataSet1 = DataLoader(dataset=input1, shuffle=False, drop_last=False, batch_size=len(input1))

sample = mdn.sample

with torch.no_grad():
    for x in dataSet0:
        outputs = model(x)
        gened_data0[last_input[0]] = sample(outputs).numpy()
    
    for x in dataSet1:
        outputs = model(x)
        gened_data1[last_input[0]] = sample(outputs).numpy()
print("model_{} used!".format('final'))

#%%
# transform the output into intergers
min = scaler.transform(np.array([0]).reshape(-1,1))[0][0]
max = scaler.transform(np.array([35]).reshape(-1,1))[0][0]

gened_data0[last_input[0]] = scaler.inverse_transform(np.clip(gened_data0[last_input[0]].values.reshape(-1,1), min, max)).round()
gened_data1[last_input[0]] = scaler.inverse_transform(np.clip(gened_data1[last_input[0]].values.reshape(-1,1), min, max)).round()


#%%
# restrict the last variable into original types
for feature in ['absences']:
    label0 = list(data_origin[data_origin.sex=='F'].absences.value_counts().keys())
    cur_col0 = torch.tensor(gened_data0.absences).unsqueeze(1)

    cur_col0 = cur_col0.expand(cur_col0.shape[0], len(label0))
    label0 = torch.tensor(label0).expand(cur_col0.shape[0], len(label0))

    diff0 = torch.argmin(torch.abs(cur_col0 - label0), dim=1).numpy()
    # gened_data0[feature] = label0[0][diff0].reshape(-1,1)
    gened_data0.loc[:,feature] = label0[0][diff0].numpy()


    label1 = list(data_origin[data_origin.sex=='M'].absences.value_counts().keys())
    cur_col1 = torch.tensor(gened_data1.absences).unsqueeze(1)

    cur_col1 = cur_col1.expand(cur_col1.shape[0], len(label1))
    label1 = torch.tensor(label1).expand(cur_col1.shape[0], len(label1))
    diff1 = torch.argmin(torch.abs(cur_col1 - label1), dim=1).numpy()
    # gened_data1[feature] = label0[0][diff1].reshape(-1,1)
    gened_data1.loc[:,feature] = label1[0][diff1].numpy()


#%%
# save data
gened_data0.to_csv("{}/interventional0_data_gened_.csv".format(dir), sep=',', index=False, header=True)
gened_data1.to_csv("{}/interventional1_data_gened_.csv".format(dir), sep=',', index=False, header=True)

