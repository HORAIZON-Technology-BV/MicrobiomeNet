# %%
import os
import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit,GridSearchCV
from tqdm import tqdm
from xgboost import XGBClassifier as xgb


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

# %% [markdown]
# Here the class labels and feature names are defined.

# %%
y = np.asarray( metadata.disease != 'healthy',dtype=int)
feature_names = microbiome.columns

# %%
np.sum(y)

# %%
param_grid = {'max_depth': [2, 5, 7],
    'learning_rate': [0.01, 0.1],
    'n_estimators': [100, 300, 800, 1000, ],
    'min_child_weight': [1, 5, ],
    'gamma': [0.5, 2],
    'subsample': [0.5, 0.6, 0.8],
    'colsample_bytree': [0.6, 0.8, 1.0],
    }

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

# %% [markdown]
# Optimize the parameters with optuna, the more trials the better. This is of course a trade-off with computational time.

# %%
sss1 = StratifiedShuffleSplit(n_splits=50,test_size=0.2, random_state=42)

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

    model = xgb(n_jobs=-1)
    
    search=GridSearchCV(
        model,
        param_grid,
        scoring='roc_auc',
        cv=sss2,
        n_jobs=-1,
        verbose=0,
        refit=True)
        
    search.fit(X_temp,y_temp)

    best_model = search.best_estimator_
    y_pred_test=best_model.predict_proba(X_test)[:,1]
    
    y_pred_train = best_model.predict_proba(X_temp)[:,1]
    
    for train,val in sss2.split(X_temp,y_temp):
        X_train,y_train = X_temp.iloc[train,:],y_temp[train]
        X_val,y_val = X_temp.iloc[val,:],y_temp[val]
    
    best_model.fit(X_train,y_train)
    
    y_pred_val = best_model.predict_proba(X_val)[:,1]
    
    list_pred_test['shuffle_'+str(shuffle_counter)] = y_pred_test
    list_test['shuffle_'+str(shuffle_counter)] = y_test

    list_pred_train['shuffle_'+str(shuffle_counter)] = y_pred_train
    list_train['shuffle_'+str(shuffle_counter)] = y_temp

    list_pred_val['shuffle_'+str(shuffle_counter)] = y_pred_val
    list_val['shuffle_'+str(shuffle_counter)] = y_val
    shuffle_counter+=1

# %%
# print('\n\tAVERAGE AUC ACROSS RUNS:')
# print(np.mean(aucs),'+-',np.std(aucs))

# %%
# Convert dictionaries to DataFrames
df_pred_test = pd.DataFrame.from_dict(list_pred_test)
df_test = pd.DataFrame.from_dict(list_test)
df_pred_train = pd.DataFrame.from_dict(list_pred_train)
df_train = pd.DataFrame.from_dict(list_train)
df_pred_val = pd.DataFrame.from_dict(list_pred_val)
df_val = pd.DataFrame.from_dict(list_val)

# Define the output path
output_path = '../../data/output/class_H_XGB_'

# Save DataFrames to CSV files
df_pred_test.to_csv(output_path + 'pred_test.csv', index=False)
df_test.to_csv(output_path + 'test.csv', index=False)
df_pred_train.to_csv(output_path + 'pred_train.csv', index=False)
df_train.to_csv(output_path + 'train.csv', index=False)
df_pred_val.to_csv(output_path + 'pred_val.csv', index=False)
df_val.to_csv(output_path + 'val.csv', index=False)



