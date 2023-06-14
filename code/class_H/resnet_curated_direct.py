# %%
import os
import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset, DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %%
from custom_nets.resnet import ResNet, train_model, evaluate
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

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

    target_metadata = metadata.loc[metadata.study_name == 'HMP_2019_ibdmdb',:]
    target_microbiome = microbiome.loc[target_metadata.index,:]

    return target_metadata,target_microbiome

metadata, microbiome = get_target_data()

# %%
y = np.asarray( metadata.disease != 'healthy',dtype=int)
feature_names = microbiome.columns


# %%
def objective(trial):
    """Define the objective function"""
    residual_dropout_check = trial.suggest_categorical("residual_dropout_check", [True,False])
    residual_dropout = trial.suggest_float('residual_dropout',0,0.5)

    weight_decay_check = trial.suggest_categorical("weight_decay_check", [True,False])
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)

    lr = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)

    if residual_dropout_check:
        params = {
            'd_numerical': len(feature_names),
            'd': trial.suggest_int('d', 64, 512),
            'd_hidden_factor' : trial.suggest_int('d_hidden_factor', 1, 4),
            'n_layers' : trial.suggest_int('n_layers', 1, 8),
            'hidden_dropout' : trial.suggest_float('hidden_dropout',0,0.5),
            'residual_dropout' : residual_dropout,
            'd_out':1
        }
    else:
        params = {
            'd_numerical': len(feature_names),
            'd': trial.suggest_int('d', 64, 512),
            'd_hidden_factor' : trial.suggest_int('d_hidden_factor', 1, 4),
            'n_layers' : trial.suggest_int('n_layers', 1, 8),
            'hidden_dropout' : trial.suggest_float('hidden_dropout',0,0.5),
            'residual_dropout' : 0,
            'd_out':1
        }
    
    model = ResNet(**params)
    model = model.to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    if weight_decay_check:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)


    model,results_dict = train_model(model, dataloaders, criterion, optimizer, dataset_sizes, phases= ['train','val'])

    return results_dict['best_val_auc']

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
    X_train = torch.tensor(np.asarray(X_train), dtype=torch.float32, device=device)
    y_train = torch.tensor(y_train, dtype=torch.float32, device=device)

    X_val = torch.tensor(np.asarray(X_val), dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val, dtype=torch.float32).to(device)

    X_test = torch.tensor(np.asarray(X_test), dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

    train_loader = DataLoader(TensorDataset(X_train,y_train), batch_size=256, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val,y_val), batch_size=256, shuffle=False)

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

    params = {
        'd_numerical':len(feature_names),
        'd':best_params['d'],
        'd_hidden_factor' : best_params['d_hidden_factor'],
        'n_layers' : best_params['n_layers'],
        'hidden_dropout' : best_params['hidden_dropout'],
        'residual_dropout' : best_params['residual_dropout'],
        'd_out' : 1,
    }

    model = ResNet(**params)
    model = model.to(device)
    criterion = torch.nn.BCEWithLogitsLoss()

    if best_params['weight_decay_check']:
        optimizer = torch.optim.AdamW(model.parameters(), lr=best_params['learning_rate'], weight_decay=best_params['weight_decay'])
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=best_params['learning_rate'])    

    model,results_dict = train_model(model, dataloaders, criterion, optimizer, dataset_sizes, phases= ['train','val'])
    
    test_auc = evaluate(model,X_test,y_test)
    aucs.append(test_auc)
    print(np.mean(aucs),'+-',np.std(aucs))

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
print('\n\tAVERAGE AUC ACROSS RUNS:')
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
output_path = '../../data/output/class_H_D_'

# Save DataFrames to CSV files
df_pred_test.to_csv(output_path + 'pred_test.csv', index=False)
df_test.to_csv(output_path + 'test.csv', index=False)
df_pred_train.to_csv(output_path + 'pred_train.csv', index=False)
df_train.to_csv(output_path + 'train.csv', index=False)
df_pred_val.to_csv(output_path + 'pred_val.csv', index=False)
df_val.to_csv(output_path + 'val.csv', index=False)



