#%%
import pandas as pd

path = "PATH_TO_DATA"
filepath = path + "\\data\\observational_data.csv"

df = pd.read_csv(filepath, sep=';')

df['Grade'] = (df['G1']+df['G2']+df['G3'])/3
df.drop(['G1','G2','G3'], axis=1, inplace=True)

df = pd.get_dummies(df, drop_first=True)

df[df['sex_M']==0].to_csv(path + "\\data\\interventional0_data_truth.csv", sep=';', index=False)

df[df['sex_M']==1].to_csv(path + "\\data\\interventional1_data_truth.csv", sep=';', index=False)


#%%
import pandas as pd

# split true intervened data
dir = "PATH_TO_DATA"
observational_data_path = "{}/observational_data.csv".format(dir)
data = pd.read_csv(observational_data_path, delimiter=',')

data[data['sex_M']==0].to_csv("{}/interventional0_data_truth.csv".format(dir), sep=',', index=False, header=True)
data[data['sex_M']==1].to_csv("{}/interventional1_data_truth.csv".format(dir), sep=',', index=False, header=True)

