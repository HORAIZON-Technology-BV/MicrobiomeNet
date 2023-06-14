# %%
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
import pickle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from tqdm import tqdm
import copy
import torchmetrics
import gc

# %%
from custom_nets.resnet import ResNet, train_model, evaluate
import optuna

# %%
microbiome = pd.read_csv('../../data/raw/curated_metagenomics/relative_abundance.csv',index_col=0).transpose()
metadata = pd.read_csv('../../data/raw/curated_metagenomics/metadata.csv',index_col='sample_id',low_memory=False)

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

y = np.asarray(metadata.disease != 'healthy',dtype=int)
feature_names = microbiome.columns

# %%
with open('../../data/raw/curated_metagenomics/resnet_params_all.pkl', 'rb') as fp:
    params = pickle.load(fp)

model = ResNet(**params)
model.load_state_dict(torch.load('../../data/raw/curated_metagenomics/resnet_curated_all.pt'))
model.eval()
model.to(device)

# %%
X_torch = torch.tensor(np.asarray(microbiome), dtype=torch.float32).cuda()
y_torch = torch.tensor(y, dtype=torch.float32).cuda()

# %%
with torch.no_grad():
    y_pred = model(X_torch.cuda())
    auroc = torchmetrics.AUROC(task="binary")
    stock_auc = auroc(y_pred, y_torch.cuda()).item()

# %%
del y_pred, auroc
torch.cuda.empty_cache()

# %%
def get_imps(i):
    cur_imps = []
    n_perms = 1000
    with torch.no_grad():
        for n in range(n_perms):
            
            # Permute column i
            X_torch_perm = copy.deepcopy(X_torch)
            permuted_tensor = X_torch_perm.permute(1, 0)  # Permute dimensions (m, n)

            # Shuffle the values in column i
            shuffled_column = torch.randperm(permuted_tensor.size(1))  # Generate random permutation
            permuted_tensor[i] = permuted_tensor[i][shuffled_column]

            # Reshape the tensor back to the original shape
            permuted_tensor = permuted_tensor.permute(1, 0)# Permute dimensions back to (n, m)
            
            y_pred_perm = model(permuted_tensor)
            
            auroc = torchmetrics.AUROC(task="binary")
            perm_auc = auroc(y_pred_perm, y_torch).item()

            cur_imps.append(stock_auc-perm_auc)
    return cur_imps

# %%

imps = {}

check = 0
with torch.no_grad():
    for i,feat in tqdm(enumerate(feature_names)):
        cur_imps = get_imps(i)
        
        imps[feat] = {
            'mean': np.mean(cur_imps),
            'sd': np.std(cur_imps)
        }
        del cur_imps

        if i % 100 == 0:
            gc.collect()
            torch.cuda.empty_cache()

# %%
imps_df = pd.DataFrame(imps).transpose()

imps_df.to_csv('../../data/output/permutation_importances.csv')


