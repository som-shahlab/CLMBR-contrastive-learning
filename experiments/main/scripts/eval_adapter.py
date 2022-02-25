import os
import shutil
import argparse
import pickle
import joblib
import pdb
import re
import yaml
import time

import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix as csr
from sklearn.linear_model import LogisticRegression as lr
from sklearn.model_selection import ParameterGrid
#from lightgbm import LGBMClassifier as gbm

from prediction_utils.util import str2bool
from prediction_utils.model_evaluation import StandardEvaluator

#------------------------------------
# Arg parser
#------------------------------------
parser = argparse.ArgumentParser(
    description = "Train model on selected hyperparameters"
)

parser.add_argument(
    '--clmbr_path',
    type=str,
    default='/local-scratch/nigam/projects/jlemmon/cl-clmbr/experiments/main/artifacts/models/clmbr',
    help='Base path for the trained end-to-end model.'
)

parser.add_argument(
    '--models_path',
    type=str,
    default='/local-scratch/nigam/projects/jlemmon/cl-clmbr/experiments/main/artifacts/models',
    help='Base path for models.'
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
    '--clmbr_type',
    type=str,
    default='pretrained'
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
    default=6,
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

def load_data(model_path):
    """
    Load datasets from split csv files.
    """
    
    test_labels = pd.read_csv(f'{model_path}/labels_test.csv')
    
    with open(f'{model_path}/features_test.pkl', 'rb') as f:
		test_feat_dict = pickle.load(f)
	test_feats = pd.DataFrame(test_feat_dict)

    test_pred_ids = pd.read_csv(f'{model_path}/prediction_ids_test.csv')
    
    test_data = (test_feats.to_numpy(),test_labels.to_numpy(), test_pred_ids.to_numpy())
    
    return test_data

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

def eval_model(args, task, clmbr_hp, cl_hp=None):
	hparams_grid = get_params(args)
	
	# Contrastive learning model location depends on base CLMBR hyperparams, so for CL need to pass both hparams lists in
	if cl_hp:
		data_path = f'{args.labelled_fpath}/{task}/contrastive_learn/{args.encoder}_sz_{clmbr_hp['size']}_do_{clmbr_hp['dropout']}_lr_{clmbr_hp['lr']}_l2_{clmbr_hp['l2']}/\
					bs_{clmbr_hp['batch_size']}_lr_{clmbr_hp['lr']}_temp_{clmbr_hp['temp']}_pool_{cl_hp['pool']}'
	else:
		data_path = f'{args.labelled_fpath}/{task}/{args.clmbr_type}/{args.encoder}_sz_{hparams["size"]}_do_{hparams["dropout"]}_lr_{hparams["lr"]}_l2_{hparams["l2"]}'
	
	if cl_hp:
		model_save_fpath = f'{args.models_path}/{task}{args.model}/{task}/{args.encoder}_sz_{clmbr_hp["size"]}_do_{clmbr_hp["dropout"]}_cd_{clmbr_hp["code_dropout"]}_dd_{clmbr_hp["day_dropout"]}_lr_{clmbr_hp["lr"]}_l2_{clmbr_hp["l2"]}'/bs_{clmbr_hp['batch_size']}_lr_{clmbr_hp['lr']}_temp_{clmbr_hp['temp']}_pool_{cl_hp['pool']}'
	else:
			model_save_fpath = f'{args.models_path}/{task}{args.model}/{task}/{args.encoder}_sz_{clmbr_hp["size"]}_do_{clmbr_hp["dropout"]}_cd_{clmbr_hp["code_dropout"]}_dd_{clmbr_hp["day_dropout"]}_lr_{clmbr_hp["lr"]}_l2_{clmbr_hp["l2"]}'
		
	
    # get data
    test_data = load_data(data_path)
    X_test, y_test, test_pred_ids = test_data[0], test_data[1], test_data[2]
	#define model save path
	
	for i, hp in enumerate(hparams_grid):
		
		m = pickle.load(open(f'{model_save_fpath}/{hp['C']}/model.pkl', 'rb'))

		evaluator = StandardEvaluator()
		t1 = time.time()
		df = pd.DataFrame({
			'pred_probs':m.predict_proba(X_test)[:,1],
			'labels':y_test.flatten(),
			'task':task,
			'test_group':'test',
			'prediction_id':test_pred_ids
		})
		
		df_test_ci, df_test = evaluator.bootstrap_evaluate(
			df,
			n_boot = args.n_boot,
			n_jobs = args.n_jobs,
			strata_vars_eval=['test_group'],
			strata_vars_boot=['labels'],
			patient_id_var='prediction_id',
			return_result_df = True
		)
		print(f'took {time.time() - t1} seconds to eval test')
		os.makedirs(f"results_save_fpath/{hp['C']}",exist_ok=True)
		
		df_test['C'] = hp['C']
		df_test['model'] = args.model
		df_test['CLMBR_model'] = 'PT' if cl_hp is None else 'CL'
		df_test_ci.reset_index(drop=True).to_csv(
			f"{results_save_fpath}/{hp['C']}/test_eval.csv"
		)


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

grid = list(
	ParameterGrid(
		yaml.load(
			open(
				f"{os.path.join(args.hparams_fpath,args.encoder)}.yml",
				'r'
			),
			Loader=yaml.FullLoader
		)
	)
)

cl_grid = list(
	ParameterGrid(
		yaml.load(
			open(
				f"{os.path.join(args.hparams_fpath,'cl')}.yml",
				'r'
			),
			Loader=yaml.FullLoader
		)
	)
)

for task in tasks:
    
    print(f"task: {task}")

    # Iterate through hyperparam lists
    for i, clmbr_hp in enumerate(clmbr_grid):
		eval_model(args, task, clmbr_hp)
        for j, cl_hp in enumerate(cl_grid):
			eval_model(args, task, clmbr_hp, cl_hp)
        
        
        

  