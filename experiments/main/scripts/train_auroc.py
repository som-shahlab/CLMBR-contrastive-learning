import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.stats import pointbiserialr
from ast import literal_eval
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.pyplot import savefig
from matplotlib.ticker import FormatStrFormatter

import seaborn as sns
sns.set_style("ticks")
sns.set_context(context='paper',font_scale=1.2)
sns.despine()

import yaml
import os
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')


model_path = '/local-scratch/nigam/projects/jlemmon/cl-clmbr/experiments/main/artifacts/models/clmbr'
results_path = '/local-scratch/nigam/projects/jlemmon/cl-clmbr/experiments/main/artifacts/results'
figure_path = '/local-scratch/nigam/projects/jlemmon/cl-clmbr/experiments/main/notebooks/figures'


tv_df = pd.read_csv(f'{model_path}/contrastive_learn/models/gru_sz_800_do_0.1_cd_0_dd_0_lr_0.001_l2_0.01/bs_2000_lr_3e-5_temp_0.01_pool_trivial/train_preds.csv')
preds = list(tv_df['preds'])
l = []
for p in preds:
    s = ' '.join(p.split())
    s = s.replace('[ ','')
    s = s.replace('[','')
    s = s.replace('   ',' ')
    s = s.replace('  ',' ')
    s = s.replace(' ]','')
    s = s.replace(']','')
    s = s.replace('\n', '')
    s = s.split(' ')
    s = [float(x) for x in s]
    l.append(softmax(s))
tv_df['preds'] = l
bin_lbls = []
logits = []
preds = list(tv_df['preds'])
lbls = list(tv_df['labels'])

epochs = list(tv_df['epoch'].unique())
auroc = []
for e in epochs:
    df = tv_df.query('epoch == @e')
    preds = list(df['preds'])
    lbls = list(df['labels'])
    sum_auc = []
    start_indices = [i for i in range(len(lbls)) if lbls[i] == 0]
    for i, si in enumerate(start_indices):
        if i < len(start_indices)-1:
            sim_matrix = preds[si:start_indices[i+1]]
            if len(sim_matrix) > 2:
                auc = roc_auc_score(lbls[si:start_indices[i+1]],sim_matrix, multi_class='ovo')
        else:
            sim_matrix = preds[si:]
            auc = roc_auc_score(lbls[si:],sim_matrix, multi_class='ovo')
        sum_auc.append(auc)
    auroc.append(np.mean(sum_auc))
    print(auroc[-1])
sns.lineplot(list(np.arange(1,len(epochs)+1)),auroc)
plt.xticks(list(np.arange(1, 21)))
plt.axvline(x=18, ls='--', color='black')
plt.savefig(f'{figure_path}/trivial_train_auroc_no_bin.png',bbox_inches='tight')