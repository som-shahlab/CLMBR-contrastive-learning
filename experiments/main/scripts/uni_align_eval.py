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
import scipy.stats as ss

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
    '--batch_size',
    type=int,
    default=128,
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
    "--iterations",
    type=int,
    default=1000,
    help="number of iterations to do when evaluating metrics"
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

def load_data(args, clmbr_hp):
    """
    Load datasets from split csv files.
    """
    data_path = f'{args.labelled_fpath}/hospital_mortality/pretrained/{args.encoder}_sz_{clmbr_hp["size"]}_do_{clmbr_hp["dropout"]}_lr_{clmbr_hp["lr"]}_l2_{clmbr_hp["l2"]}'
    
    val_pids = pd.read_csv(f'{data_path}/ehr_ml_patient_ids_val.csv')
    test_pids = pd.read_csv(f'{data_path}/ehr_ml_patient_ids_test.csv')
    
    val_days = pd.read_csv(f'{data_path}/day_indices_val.csv')
    test_days = pd.read_csv(f'{data_path}/day_indices_test.csv')
    
    val_labels = pd.read_csv(f'{data_path}/labels_val.csv')
    test_labels = pd.read_csv(f'{data_path}/labels_test.csv')
    
    val_data = (val_labels.to_numpy().flatten(),val_pids.to_numpy().flatten(),val_days.to_numpy().flatten())
    test_data = (test_labels.to_numpy().flatten(),test_pids.to_numpy().flatten(),test_days.to_numpy().flatten())
    
    return val_data, test_data

def eval_alignment(args, model, data):
	align_list = []
	for i in range(args.iterations):
		batch = np.random.choice(data, args.batch_size)
		print(batch)
		z1 = model(batch)
		z2 = model(batch)
		
		align = alignment(z1, z2)
		print(align)
		align_list.append(align)
	return np.array(align_list)

def eval_uniform(args, model, data):
	uniform_list = []
	for i in range(args.iterations):
		batch = np.random.choice(data, args.batch_size)
		print(batch)
		x = model(batch)
		
		uniform = uniformity(x)
		print(uniform)
		uniform_list.append(uniform)
	return np.array(uniform_list)

def alignment(x, y, alpha=2):
	return (x - y).norm(p=2, dim=1).pow(alpha).mean()

def uniformity(x, t=2):
	return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

def compute_ci(vals, metric,  a=5.0):
	lower_p = a/2
	upper_p = (100 - a) + a/2
	
	lower = max(0.0, np.percentile(vals, lower_p))
	med = np.median(vals)
	upper = min(1.0, np.percentile(vals, upper_p))
	
	df = pd.DataFrame({'metric':metric, 'lower_ci':lower, 'med_ci':med, 'upper_ci':ci})
	
	return df

def eval_model(args, task, val_dataset, test_dataset, clmbr_hp, cl_hp=None):
	
	if cl_hp:
		clmbr_model_path = f'{args.clmbr_path}/contrastive_learn/models/{args.encoder}_sz_{clmbr_hp["size"]}_do_{clmbr_hp["dropout"]}_cd_{clmbr_hp["code_dropout"]}_dd_{clmbr_hp["day_dropout"]}_lr_{clmbr_hp["lr"]}_l2_{clmbr_hp["l2"]}/bs_{cl_hp['batch_size']}_lr_{clmbr_hp['lr']}_temp_{cl_hp['temp']}_pool_{cl_hp['pool']}'
	else:
		clmbr_model_path = f'{args.clmbr_path}/pretrained/models/{args.encoder}_sz_{clmbr_hp["size"]}_do_{clmbr_hp["dropout"]}_cd_{clmbr_hp["code_dropout"]}_dd_{clmbr_hp["day_dropout"]}_lr_{clmbr_hp["lr"]}_l2_{clmbr_hp["l2"]}'
	
	
	if cl_hp:
		results_save_fpath = f'{args.results_path}/contrastive_learn/{task}/{args.encoder}_sz_{clmbr_hp["size"]}_do_{clmbr_hp["dropout"]}_cd_{clmbr_hp["code_dropout"]}_dd_{clmbr_hp["day_dropout"]}_lr_{clmbr_hp["lr"]}_l2_{clmbr_hp["l2"]}/bs_{cl_hp['batch_size']}_lr_{cl_hp['lr']}_temp_{cl_hp['temp']}_pool_{cl_hp['pool']}'
	else:
		results_save_fpath = f'{args.results_path}/pretrained/{task}/{args.encoder}_sz_{clmbr_hp["size"]}_do_{clmbr_hp["dropout"]}_cd_{clmbr_hp["code_dropout"]}_dd_{clmbr_hp["day_dropout"]}_lr_{clmbr_hp["lr"]}_l2_{clmbr_hp["l2"]}
		
		
	
	#define model save path	
	clmbr_model = ehr_ml.clmbr.CLMBR.from_pretrained(clmbr_model_path, args.device).to(args.device)
	# set model to train so dropout is activated
	clmbr_model.train()
	config = clmbr_model.clmbr_model.config
	
	val_loader = DataLoader(val_dataset, config['num_first'], is_val=False, batch_size=args.batch_size, seed=args.seed, device=args.device)
	test_loader = DataLoader(test_dataset, config['num_first'], is_val=False, batch_size=args.batch_size, seed=args.seed, device=args.device)
	
	val_data = []
	test_data = []
	
	for i in range(len(val_loader)):
		val_data.append(next(val_loader))
	for i in range(len(test_loader)):
		test_data.append(next(test_loader))
	
	val_align= eval_alignment(args, clmbr_model, val_data)
	test_align = eval_alignment(args, clmbr_model, test_data)
	
	val_uniform = eval_uniform(args, clmbr_model, val_data)
	test_uniform = eval_uniform(args, clmbr_model, test_data)
	
	ci_df = pd.Dataframe()
	
	df = compute_ci(val_align, 'alignment')
	df['split'] = 'val'
	df['CLMBR'] = 'CL' if cl_hp else 'BL'
	ci_df = pd.concat([ci_df,df])
	
	df = compute_ci(val_uniform, 'uniformity')
	df['split'] = 'val'
	df['CLMBR'] = 'CL' if cl_hp else 'BL'
	ci_df = pd.concat([ci_df,df])
	
	df = compute_ci(test_align, 'alignment')
	df['split'] = 'test'
	df['CLMBR'] = 'CL' if cl_hp else 'BL'
	ci_df = pd.concat([ci_df,df])
	
	df = compute_ci(test_uniform, 'uniformity')
	df['split'] = 'test'
	df['CLMBR'] = 'CL' if cl_hp else 'BL'
	ci_df = pd.concat([ci_df,df])
	
	ci_df.reset_index(drop=True).to_csv(f'{results_save_fpath}/uni_align_eval.csv')

#-------------------------------------------------------------------
# run
#-------------------------------------------------------------------
args = parser.parse_args()

# threads
joblib.Parallel(n_jobs=args.n_jobs)

# set seed
torch.manual_seed(args.seed)

# parse tasks and train_group
tasks = ['hospital_mortality', 'LOS_7', 'icu_admission', 'readmission_30']

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

val_dataset, test_dataset = None, None

for task in tasks:
    
    print(f"task: {task}")

    # Iterate through hyperparam lists
    for i, clmbr_hp in enumerate(clmbr_grid):
		# all models use same dataset so only load once
		if i == 0:
			clmbr_model_path = f'{args.clmbr_path}/pretrained/models/{args.encoder}_sz_{clmbr_hp["size"]}_do_{clmbr_hp["dropout"]}_cd_{clmbr_hp["code_dropout"]}_dd_{clmbr_hp["day_dropout"]}_lr_{clmbr_hp["lr"]}_l2_{clmbr_hp["l2"]}'
			val_data, test_data = load_data(args, clmbr_hp)

			val_dataset = PatientTimelineDataset(args.extract_path + '/extract.db', 
									 args.extract_path + '/ontology.db', 
									 f'{clmbr_model_path}/info.json', 
									 val_data, 
									 val_data ).to(args.device)

			test_dataset = PatientTimelineDataset(args.extract_path + '/extract.db', 
							 args.extract_path + '/ontology.db', 
							 f'{clmbr_model_path}/info.json', 
							 test_data, 
							 test_data ).to(args.device)
		
		eval_model(args, task, val_dataset, test_dataset, clmbr_hp)
        for j, cl_hp in enumerate(cl_grid):
			eval_model(args, task, val_dataset, test_dataset, clmbr_hp, cl_hp)
        
        
        

  