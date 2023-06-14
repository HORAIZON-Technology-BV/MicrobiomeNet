# %%
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
from custom_nets.resnet import ResNet, train_model, evaluate
import optuna

# %% [markdown]
# ## Pre-processing
# 
# This is all some quite elaborate steps to load in the data. It definetely does not need to be as difficult as this. The main goal is to have a train, val and test set of features and labels. As long as you have that all correctly defined it should all work a-ok.

# %%
microbiome = pd.read_csv('../../data/raw/curated_metagenomics/relative_abundance.csv',index_col=0).transpose()
metadata = pd.read_csv('../../data/raw/curated_metagenomics/metadata.csv',index_col='sample_id',low_memory=False)

# %% [markdown]
# For this example we will try to classify disease from healthy based on microbiome. Disease is classified as diseased (according to the original data) and BMI<16 | BMI=>30. These are the boundaries of severe underweight and obesity.

# %%
#get stool samples
metadata = metadata.loc[metadata.body_site == 'stool',:]

#Add obesity disease tags to disease BMI
to_change = metadata.BMI>=30
metadata.loc[to_change,'disease'] = 'obesity'

to_change = metadata.BMI<16
metadata.loc[to_change,'disease'] = 'severe_underweight'

# Remove all disease NaNs
metadata = metadata.loc[metadata.disease==metadata.disease,:]

#Take only the adults
to_keep = metadata.age_category != 'newborn'
metadata = metadata.loc[to_keep,:]

# Get the overlapping set of samples between metadata and microbiome data
overlapping_samples = list(set(metadata.index) & set(microbiome.index))
microbiome= microbiome.loc[overlapping_samples,:]
metadata = metadata.loc[overlapping_samples,:]


# %% [markdown]
# Here the class labels and feature names are defined.

# %%
y = np.asarray( metadata.disease != 'healthy',dtype=int)
feature_names = microbiome.columns

# %% [markdown]
# ## Split data
# 
# Here we split the data into the train val and test sets. Since the curated set is rather big, we stick to 2000 test samples and 1000 validation samples. This can of course be tuned.
# 
# After splitting the data is transformed to tensors which are moved to the device, the train code assumes all tensors and moves are already moved to the GPU, this is beneficial as it speeds up the loading of the data a lot. But, it is something to be mindful about.

# %%
X_train, X_val, y_train, y_val = train_test_split(microbiome,y, test_size=1000)

#Transfer to tensors and bring to device
X_train = torch.tensor(np.asarray(X_train), dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)

X_val = torch.tensor(np.asarray(X_val), dtype=torch.float32).to(device)
y_val = torch.tensor(y_val, dtype=torch.float32).to(device)

# %%
train_loader = DataLoader( TensorDataset(X_train,y_train), batch_size=256, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val,y_val), batch_size=256, shuffle=True,)

dataloaders = {
    'train' : train_loader,
    'val' : val_loader,
}

dataset_sizes = {
    'train' : len(y_train),
    'val' : len(y_val)
}

# %% [markdown]
# ## Define the objective function for optuna
# 
# This is where we define the objective function to be optimized by optuna. This code is also adapted from the sample example as the ResNet code. The trial variables are all optimized in order to maximize validation AUC.
# 

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

# %% [markdown]
# Optimize the parameters with optuna, the more trials the better. This is of course a trade-off with computational time.

# %%
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=500)

# %% [markdown]
# ## Retrain the model with the best params
# You might recognize this from the objective function, essentially this is the same code, but it will pick the best params found by optuna.

# %%
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

# %%
import pickle
with open('../../data/raw/curated_metagenomics/resnet_params_all.pkl', 'wb') as fp:
    pickle.dump(params, fp)

# %%
torch.save(model.state_dict(), '../../data/raw/curated_metagenomics/resnet_curated_all.pt')


