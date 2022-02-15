import os
import shutil
import argparse
import pickle
import joblib
import pdb
import re
import yaml

import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix as csr
from sklearn.linear_model import LogisticRegression as lr
from sklearn.model_selection import ParameterGrid
#from lightgbm import LGBMClassifier as gbm

from prediction_utils.pytorch_utils.metrics import StandardEvaluator
from prediction_utils.util import str2bool

#------------------------------------
# Arg parser
#------------------------------------
parser = argparse.ArgumentParser(
    description = "Train model on selected hyperparameters"
)

parser.add_argument(
    '--model_path',
    type=str,
    default='/local-scratch/nigam/projects/jlemmon/cl-clmbr/experiments/main/artifacts/models/clmbr',
    help='Base path for the trained end-to-end model.'
)

parser.add_argument(
    '--results_path',
    type=str,
    default='/local-scratch/nigam/projects/jlemmon/cl-clmbr/experiments/main/artifacts/results/clmbr',
    help='Base path for the evaluation results.'
)

parser.add_argument(
    '--extract_path',
    type=str,
    default='/local-scratch/nigam/projects/jlemmon/cl-clmbr/experiments/main/data/extracts/20210723',
    help='Base path for the extracted database.'
)

parser.add_argument(
    '--cohort_fpath',
    type=str,
    default="/local-scratch/nigam/projects/jlemmon/cl-clmbr/experiments/main/data/cohort",
    help='Base path for cohort file'
)

parser.add_argument(
    '--hparams_fpath',
    type=str,
    default="/local-scratch/nigam/projects/jlemmon/cl-clmbr/experiments/main/hyperparams",
    help='Base path for hyperparameter files'
)

parser.add_argument(
    '--labelled_fpath',
    type=str,
    default="/local-scratch/nigam/projects/jlemmon/cl-clmbr/experiments/main/data/labelled_data",
    help='Base path for labelled data directory'
)

parser.add_argument(
    "--con_learn",
    action='store_true',
    help='Whether to use base pretrained CLMBR or CLMBR finetuned with Contrastive Learning.'
)

parser.add_argument(
    "--encoder",
    type=str,
    default='gru',
    help='gru/transformer',
)

parser.add_argument(
    "--model",
    type=str,
    default="lr"
)

parser.add_argument(
    '--size',
    type=int,
    default=800,
    help='Size of representation vector.'
)

parser.add_argument(
    '--dropout',
    type=float,
    default=0.1,
    help='Dropout proportion for training.'
)

parser.add_argument(
    '--lr',
    type=float,
    default=0.01,
    help='Learning rate for training.'
)

parser.add_argument(
    "--n_jobs",
    type=int,
    default=4,
    help="number of threads"
)

parser.add_argument(
    "--seed",
    type = int,
    default = 44,
    help = "seed for deterministic training"
)

parser.add_argument(
    "--n_searches",
    type=int,
    default=100,
    help="number of random searches to conduct for hparam search"
)

parser.add_argument(
    "--overwrite",
    type = str2bool,
    default = "false",
    help = "whether to overwrite existing artifacts",
)


#-------------------------------------------------------------------
# helper functions
#-------------------------------------------------------------------

def load_data(args, task):
    """
    Load datasets from split csv files.
    """
    
    if args.con_learn:
        model_path = f'{args.labelled_fpath}/con_learn/{args.encoder}_{args.size}_{args.dropout}/{task}'
    else:
        model_path = f'{args.labelled_fpath}/{args.encoder}_{args.size}_{args.dropout}/{task}'
    
    train_labels = pd.read_csv(f'{model_path}/labels_train.csv')
    val_labels = pd.read_csv(f'{model_path}/labels_val.csv')
    test_labels = pd.read_csv(f'{model_path}/labels_test.csv')
    
    train_feats = pd.read_csv(f'{model_path}/features_train.csv')
    val_feats = pd.read_csv(f'{model_path}/features_val.csv')
    test_feats = pd.read_csv(f'{model_path}/features_test.csv')
    
    train_pred_ids = pd.read_csv(f'{model_path}/prediction_ids_train.csv')
    val_pred_ids = pd.read_csv(f'{model_path}/prediction_ids_val.csv')
    test_pred_ids = pd.read_csv(f'{model_path}/prediction_ids_test.csv')
    
    train_data = (train_feats.to_numpy(),train_labels.to_numpy(), train_pred_ids.to_numpy())
    val_data = (val_feats.to_numpy(), val_labels.to_numpy(), val_pred_ids.to_numpy())
    
    return train_data, val_data

def get_params(args):
    param_grid = yaml.load(
        open(
            os.path.join(
                args.hparams_fpath,
                f'{args.model}.yml'
            ), 'r'
        ),
        Loader = yaml.FullLoader
    )
    
    param_grid = list(ParameterGrid(param_grid))
    np.random.shuffle(param_grid)
    
    if args.n_searches < len(param_grid):
        param_grid = param_grid[:args.n_searches]
    
    return param_grid

#-------------------------------------------------------------------
# run
#-------------------------------------------------------------------
args = parser.parse_args()

# threads
joblib.Parallel(n_jobs=args.n_jobs)

# set seed
np.random.seed(args.seed)

# parse tasks and train_group
tasks = ['hospital_mortality', 'LOS_7', 'icu_admission', 'readmission_30']

# initialize evaluator
evaluator = StandardEvaluator(metrics=['loss_bce'])

for task in tasks:
    
    print(f"task: {task}")
    
    hparams_grid = get_params(args)

    # get data
    train_data, val_data = load_data(args, task)
    X_train, y_train, train_pred_ids = train_data[0], train_data[1], train_data[2]
    X_val, y_val, val_pred_ids = val_data[0], val_data[1], val_data[2]
    
    # Iterate through hyperparam list
    for i, hparams in enumerate(hparams_grid):
        print(hparams)
        
        #define model save path
        model_name = '_'.join([
            args.model,
            f'{args.encoder}',
            f'{args.size}',
            f'{args.dropout}'
        ])
        
        model_num = str(i)
        
        fpath = os.path.join(
            args.model_path,
            f"{'contrastive_learn' if args.con_learn else 'baseline'}",
            task,
            'models',
            model_name,
            model_num
        )

        os.makedirs(fpath,exist_ok=True)
        
        #train model
        if args.model == 'lr':
            m = lr(
                n_jobs=args.n_jobs,
                **hparams
            )
            m.fit(X_train, y_train.flatten())
        else:
            pass
        
        #get prediction probability df with validation data
        df = pd.DataFrame({
            'pred_probs':m.predict_proba(X_val)[:,1],
            'labels':y_val.flatten(),
            'task':task,
            'prediction_id':val_pred_ids
        })
        
        #save model, hparams used and validation pred probs
        pickle.dump(
            m,
            open(f'{fpath}/model.pkl', 'wb')
        )
        
        yaml.dump(
            hparams,
            open(f'{fpath}/hparams.yml', 'w')
        )
        
        fpath = os.path.join(
            args.results_path,
            f"{'contrastive_learn' if args.con_learn else 'baseline'}",
            task,
            'results',
            model_name,
            model_num
        )
            
        os.makedirs(fpath,exist_ok=True)
            
        df.reset_index(drop=True).to_csv(
            f'{fpath}/val_pred_probs.csv'
        )
        

  