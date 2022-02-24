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

from prediction_utils.model_evaluation import StandardEvaluator
from prediction_utils.util import str2bool

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
    "--n_boot",
    type=int,
    default=10000,
    help="number of bootstrap iterations for evaluation"
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

	train_labels = pd.read_csv(f'{model_path}/labels_train.csv')
	val_labels = pd.read_csv(f'{model_path}/labels_val.csv')
	
	with open('f{model_path}/features_train.pkl', 'rb') as f:
		train_feat_dict = pickle.load(f)
	train_feats = pd.DataFrame(train_feat_dict)
	
	with open('f{model_path}/features_val.pkl', 'rb') as f:
		val_feat_dict = pickle.load(f)
	val_feats = pd.DataFrame(val_feat_dict)

	train_pred_ids = pd.read_csv(f'{model_path}/prediction_ids_train.csv')
	val_pred_ids = pd.read_csv(f'{model_path}/prediction_ids_val.csv')

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

def train_model(args, task, clmbr_hp, cl_hp, X_train, y_train, X_val, y_val, val_pred_ids, X_test, y_test, test_pred_ids):
	hparams_grid = get_params(args)
	
	data_path = f"{args.labelled_fpath}/{task}/contrastive_learn/{args.encoder}_sz_{hparams["size"]}_do_{hparams["dropout"]}_cd_{hparams["code_dropout"]}_dd_{hparams["day_dropout"]}_lr_{hparams["lr"]}_l2_{hparams["l2"]}/bs_{cl_hp['batch_size']}_lr_{cl_hp['lr']}_temp_{cl_hp['temp']}_pool_{cl_hp['pool']}"
	
	results_save_fpath = f"{args.results_path}/{args.model}/{task}/{args.encoder}_sz_{hparams["size"]}_do_{hparams["dropout"]}_cd_{hparams["code_dropout"]}_dd_{hparams["day_dropout"]}_lr_{hparams["lr"]}_l2_{hparams["l2"]}/bs_{cl_hp['batch_size']}_lr_{cl_hp['lr']}_temp_{cl_hp['temp']}_pool_{cl_hp['pool']}"
		
	#define model save path
	
	lr_eval_df = pd.DataFrame()
	
	for i, hp in enumerate(hparams_grid):
		model_name = '_'.join([
			args.model,
			f'{hp["C"]}',
		])

		model_num = str(i)


		os.makedirs(f"{fpath}/{hp['C']}",exist_ok=True)

		#train model
		if args.model == 'lr':
			m = lr(n_jobs=args.n_jobs, **hp)
		else:
			pass
		
		m.fit(X_train, y_train.flatten())

		#get prediction probability df with validation data
		df = pd.DataFrame({
			'pred_probs':m.predict_proba(X_val)[:,1],
			'labels':y_val.flatten(),
			'task':task,
			'prediction_id':val_pred_ids,
			'test_group':'val',
			'C':hp['C'],
			'model':args.model
		})
		print(df)
		#save model, hparams used and validation pred probs
		pickle.dump(
			m,
			open(f"{fpath}/{hp['C']}/model.pkl", 'wb')
		)

		yaml.dump(
			hp,
			open(f"{fpath}/{hp['C']}/hparams.yml", 'w')
		)

		df.reset_index(drop=True).to_csv(
			f"{fpath}/{hp['C']}/val_pred_probs.csv"
		)
		# initialize evaluator
		evaluator = StandardEvaluator()

		df_eval_ci, df_eval = evaluator.bootstrap_evaluate(
			df,
			n_boot = args.n_boot,
			n_jobs = args.n_jobs,
			strata_vars_eval=['test_group'],
			strata_vars_boot=['labels'],
			patient_id_var='prediction_id',
			return_result_df = True
		)
		print(df_eval_ci)
		
		print(df_eval)

		os.makedirs(f"{fpath}/{hp['C']}",exist_ok=True)

		df_eval_ci.reset_index(drop=True).to_csv(
			f"{fpath}/{hp['C']}/val_eval.csv"
		)

		df_eval['C'] = hp['C']
		df_eval['model'] = args.model
		df_eval['CLMBR_model'] = 'CL'


		lr_eval_df = pd.concat([lr_eval_df, df_eval], ignore_index=True)
		
		evaluator = StandardEvaluator()
		
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
		
		os.makedirs(f"fpath/{hp['C']}",exist_ok=True)
		
		df_test['C'] = hp['C']
		df_test['model'] = args.model
		df_test['CLMBR_model'] = 'CL'
		df_test_ci.reset_index(drop=True).to_csv(
			f"{fpath}/{hp['C']}/test_eval.csv"
		)
		
	print(clmbr_hp)
	print(lr_eval_df)

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


clmbr_grid = list(
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
		train_model(args, task, clmbr_hp)
		for j, cl_hp in enumerate(cl_grid):
			train_model(args, task, clmbr_hp, cl_hp)
        
        
        

  