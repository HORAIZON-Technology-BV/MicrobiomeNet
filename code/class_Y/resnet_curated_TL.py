# %%
import os
import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %%
from custom_nets.resnet import ResNet, train_model, evaluate
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

# %%
with open('../../data/raw/curated_metagenomics/resnet_params_YachidaS_2019.pkl', 'rb') as fp:
    params = pickle.load(fp)

model = ResNet(**params)
model.load_state_dict(torch.load('../../data/raw/curated_metagenomics/resnet_curated_YachidaS_2019.pt'))
print(model)

# %%
for name,param in model.named_parameters():
    print(name)
    # if ('head' in name) | ('last_normalization' in name):
    #     param.requires_grad = True
    # else:
    #     param.requires_grad = False

# %% [markdown]
# ## Pre-processing
# 
# This is all some quite elaborate steps to load in the data. It definetely does not need to be as difficult as this. The main goal is to have a train, val and test set of features and labels. As long as you have that all correctly defined it should all work a-ok.

# %% [markdown]
# For this example we will try to classify disease from healthy based on microbiome. Disease is classified as diseased (according to the original data) and BMI<16 | BMI=>30. These are the boundaries of severe underweight and obesity.

# %%
def get_target_data():

    microbiome = pd.read_csv('../../data/raw/curated_metagenomics/relative_abundance.csv',index_col=0).transpose()
    metadata = pd.read_csv('../../data/raw/curated_metagenomics/metadata.csv',index_col='sample_id',low_memory=False)

    #get stool samples
    metadata = metadata.loc[metadata.body_site == 'stool',:]

    #Add obesity disease tags to disease BMI
    to_change = metadata.BMI>=30
    metadata.loc[to_change,'disease'] = 'obesity'

    to_change = metadata.BMI<16
    metadata.loc[to_change,'disease'] = 'severe_underweight'

    # Remove all disease NaNs
    metadata = metadata.loc[metadata.disease==metadata.disease,:]

    #
    to_keep = metadata.age_category != 'newborn'
    metadata = metadata.loc[to_keep,:]

    # Get the overlapping set of samples between metadata and microbiome data
    overlapping_samples = list(set(metadata.index) & set(microbiome.index))
    microbiome= microbiome.loc[overlapping_samples,:]
    metadata = metadata.loc[overlapping_samples,:]

    target_metadata = metadata.loc[metadata.study_name == 'YachidaS_2019',:]
    target_microbiome = microbiome.loc[target_metadata.index,:]

    return target_metadata,target_microbiome

metadata, microbiome = get_target_data()


# %% [markdown]
# Here the class labels and feature names are defined.

# %%
y = np.asarray(metadata.disease != 'healthy',dtype=int)
feature_names = microbiome.columns
np.sum(y)

# %% [markdown]
# ## Split data
# 
# Here we split the data into the train val and test sets. Since the curated set is rather big, we stick to 2000 test samples and 1000 validation samples. This can of course be tuned.
# 
# After splitting the data is transformed to tensors which are moved to the device, the train code assumes all tensors and moves are already moved to the GPU, this is beneficial as it speeds up the loading of the data a lot. But, it is something to be mindful about.

# %% [markdown]
# ## Define the objective function for optuna
# 
# This is where we define the objective function to be optimized by optuna. This code is also adapted from the sample example as the ResNet code. The trial variables are all optimized in order to maximize validation AUC.
# 

# %%
def objective(trial):
    """Define the objective function"""

    weight_decay_check = trial.suggest_categorical("weight_decay_check", [True,False])
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    lr = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)

    with open('../../data/raw/curated_metagenomics/resnet_params_YachidaS_2019.pkl', 'rb') as fp:
        params = pickle.load(fp)
    model = ResNet(**params)
    model.load_state_dict(torch.load('../../data/raw/curated_metagenomics/resnet_curated_YachidaS_2019.pt'))

    for name,param in model.named_parameters():
        if ('head' in name) | ('last_normalization' in name) | ('first_layer' in name):
            param.requires_grad = True
        else:
            param.requires_grad = False

    model = model.to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    if weight_decay_check:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)


    model,results_dict = train_model(model, dataloaders, criterion, optimizer, dataset_sizes, phases= ['train','val'])

    return results_dict['best_val_auc']

# %% [markdown]
# Optimize the parameters with optuna, the more trials the better. This is of course a trade-off with computational time.

# %%
sss1 = StratifiedShuffleSplit(n_splits=50,test_size=0.2, random_state=42)

aucs = []
list_pred_train={}
list_train={}

list_pred_val={}
list_val={}

list_pred_test={}
list_test={}

shuffle_counter = 0
for temp, test in tqdm(sss1.split(microbiome,y)):
    X_temp, y_temp = microbiome.iloc[temp,:], y[temp]
    X_test, y_test = microbiome.iloc[test,:], y[test]

    sss2 = StratifiedShuffleSplit(n_splits=1,test_size=0.2, random_state=42)

    for train,val in sss2.split(X_temp,y_temp):
        X_train, y_train = microbiome.iloc[train,:], y[train]
        X_val, y_val = microbiome.iloc[val,:], y[val]

    #Transfer to tensors and bring to device
    X_train = torch.tensor(np.asarray(X_train), dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)

    X_val = torch.tensor(np.asarray(X_val), dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val, dtype=torch.float32).to(device)

    X_test = torch.tensor(np.asarray(X_test), dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

    train_loader = DataLoader( TensorDataset(X_train,y_train), batch_size=256, shuffle=True)
    val_loader = DataLoader( TensorDataset(X_val,y_val), batch_size=256, shuffle=False,)

    dataloaders = {
        'train' : train_loader,
        'val' : val_loader,
    }

    dataset_sizes = {
        'train' : len(y_train),
        'val' : len(y_val)
    }

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)


    best_params = study.best_params

    with open('../../data/raw/curated_metagenomics/resnet_params_YachidaS_2019.pkl', 'rb') as fp:
        params = pickle.load(fp)
    model = ResNet(**params)
    model.load_state_dict(torch.load('../../data/raw/curated_metagenomics/resnet_curated_YachidaS_2019.pt'))

    for name,param in model.named_parameters():
        if ('head' in name) | ('last_normalization' in name) | ('first_layer' in name):
            param.requires_grad = True
        else:
            param.requires_grad = False

    model = model.to(device)
    criterion = torch.nn.BCEWithLogitsLoss()

    if best_params['weight_decay_check']:
        optimizer = torch.optim.AdamW(model.parameters(), lr=best_params['learning_rate'], weight_decay=best_params['weight_decay'])
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=best_params['learning_rate'])    

    model,results_dict = train_model(model, dataloaders, criterion, optimizer, dataset_sizes, phases= ['train','val'])


    model.eval()
    test_auc = evaluate(model,X_test,y_test)
    aucs.append(test_auc)

    y_pred_test=model(X_test).detach().cpu().numpy()
    y_test=y_test.detach().cpu().numpy()
    
    y_pred_train = model(X_train).detach().cpu().numpy()
    y_train=y_train.detach().cpu().numpy()
    
    y_pred_val = model(X_val).detach().cpu().numpy()
    y_val=y_val.detach().cpu().numpy()
    
    list_pred_test['shuffle_'+str(shuffle_counter)] = y_pred_test
    list_test['shuffle_'+str(shuffle_counter)] = y_test

    list_pred_train['shuffle_'+str(shuffle_counter)] = y_pred_train
    list_train['shuffle_'+str(shuffle_counter)] = y_train

    list_pred_val['shuffle_'+str(shuffle_counter)] = y_pred_val
    list_val['shuffle_'+str(shuffle_counter)] = y_val
    shuffle_counter+=1


# %%
print(np.mean(aucs),'+-',np.std(aucs))

# %%
# Convert dictionaries to DataFrames
df_pred_test = pd.DataFrame.from_dict(list_pred_test)
df_test = pd.DataFrame.from_dict(list_test)
df_pred_train = pd.DataFrame.from_dict(list_pred_train)
df_train = pd.DataFrame.from_dict(list_train)
df_pred_val = pd.DataFrame.from_dict(list_pred_val)
df_val = pd.DataFrame.from_dict(list_val)

# Define the output path
output_path = '../../data/output/class_Y_TL_'

# Save DataFrames to CSV files
df_pred_test.to_csv(output_path + 'pred_test.csv', index=False)
df_test.to_csv(output_path + 'test.csv', index=False)
df_pred_train.to_csv(output_path + 'pred_train.csv', index=False)
df_train.to_csv(output_path + 'train.csv', index=False)
df_pred_val.to_csv(output_path + 'pred_val.csv', index=False)
df_val.to_csv(output_path + 'val.csv', index=False)


