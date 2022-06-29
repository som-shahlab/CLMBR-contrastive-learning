import os
import math
import shutil
import argparse
import pickle
import joblib
import pdb
import re
import yaml
import time
import torch
import random

import pandas as pd
import numpy as np
import scipy.stats as ss

from scipy.sparse import csr_matrix as csr
from sklearn.linear_model import LogisticRegression as lr
from sklearn.model_selection import ParameterGrid
#from lightgbm import LGBMClassifier as gbm

from prediction_utils.util import str2bool
# from prediction_utils.model_evaluation import StandardEvaluator

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import ehr_ml.timeline
import ehr_ml.ontology
import ehr_ml.index
import ehr_ml.labeler
import ehr_ml.clmbr
from ehr_ml.clmbr import Trainer
from ehr_ml.clmbr import PatientTimelineDataset
from ehr_ml.clmbr.dataset import DataLoader

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
    '--ft_model_path',
    type=str,
    default='/local-scratch/nigam/projects/jlemmon/cl-clmbr/experiments/main/artifacts/models/clmbr/contrastive_learn/models/gru_sz_800_do_0.1_cd_0_dd_0_lr_0.001_l2_0.01/best',
    help='Base path for the best finetuned model.'
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
    default=64,
    help='Number of patients in batch'
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
    default=2000,
    help="number of iterations to do when evaluating metrics"
)

parser.add_argument(
    "--overwrite",
    type = str2bool,
    default = "false",
    help = "whether to overwrite existing artifacts",
)

parser.add_argument(
    '--pooler',
    type=str,
    default='mean_rep',
    help='Pooling method to get representations.'
)

parser.add_argument(
    '--device',
    type=str,
    default='cuda:0',
    help='Device to run torch model on.'
)
#-------------------------------------------------------------------
# classes
#-------------------------------------------------------------------
class MLPLayer(nn.Module):
	"""
	Linear classifier layer for use with [CLS] pooler.
	"""
	def __init__(self, size):
		super().__init__()
		self.dense = nn.Linear(size, size)
		self.activation = nn.Tanh()
		
	def forward(self, features):
		x = self.dense(features)
		x = self.activation(x)
		
		return x

class Pooler(nn.Module):
	"""
	Pooler module to get the pooled embedding value. [cls] is equivalent to the final hidden layer embedding.
	"""
	def __init__(self, pooler, device=None):
		super().__init__()
		self.pooler = pooler
		self.device = device if device is not None else 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
	def forward(self, embeds, day_indices=None):
		# Only CLS style pooling for now
		if self.pooler == 'cls':
			return embeds[-1]
		elif self.pooler == 'mean_rep':
			if len(embeds) == 2:
				embeds = embeds[0]
			return torch.mean(embeds,1,True)
		elif self.pooler == 'rand_day':
			outputs = torch.tensor([]).to(self.device)
			for i, e in enumerate(embeds):
				outputs = torch.concat((outputs, e[day_indices[i]]), 0)
			outputs = torch.reshape(outputs, (embeds.shape[0], 1, embeds.shape[-1]))
			return outputs
		else:
			return embeds

class MetricWrapper(nn.Module):
	"""
	Wrapper to get representations for 
	"""
	def __init__(self, clmbr_model, pooler, device=None):
		super().__init__()
		self.timeline_model = clmbr_model.timeline_model
		self.config = clmbr_model.config
		self.pooler = Pooler(pooler, device)
		self.device = device if device is not None else 'cuda:0' if torch.cuda.is_available() else 'cpu'
		self.linear = MLPLayer(clmbr_model.config['size']).to(self.device)

	def forward(self, batch):
		outputs = dict()
		# For patient timeline in batch get CLMBR embedding
		z1_embeds = self.timeline_model(batch["rnn"])
		# print(z1_embeds.shape)
		# Run batch through CLMBR again to get different masked embedding for positive pairs
		z2_embeds = self.timeline_model(batch["rnn"])
		
		rand_day_indices = None
		if self.pooler.pooler == 'rand_day':
			rand_day_indices = []
			for di in batch['day_index']:
				rand_day_indices.append(random.choice(di))
		# Use pooler to get target embeddings
		z1_target_embeds = self.pooler(z1_embeds, rand_day_indices)
		z2_target_embeds = self.pooler(z2_embeds, rand_day_indices)
		
		# pooled_embeds = self.linear(pooled_embeds)
		z1 = self.linear(z1_target_embeds)
		z2 = self.linear(z2_target_embeds)

		outputs['z1'] = z1.view(len(batch['pid']), z1.size(-1))
		outputs['z2'] = z2.view(len(batch['pid']), z2.size(-1))
		return outputs

#-------------------------------------------------------------------
# helper functions
#-------------------------------------------------------------------

def load_data(args, clmbr_hp):
	"""
	Load datasets from split csv files.
	"""
	data_path = f'{args.labelled_fpath}/hospital_mortality/pretrained/{args.encoder}_sz_{clmbr_hp["size"]}_do_{clmbr_hp["dropout"]}_lr_{clmbr_hp["lr"]}_l2_{clmbr_hp["l2"]}'

	val_pids = pd.read_csv(f'{data_path}/ehr_ml_patient_ids_val.csv').to_numpy().flatten()
	test_pids = pd.read_csv(f'{data_path}/ehr_ml_patient_ids_test.csv').to_numpy().flatten()

	val_days = pd.read_csv(f'{data_path}/day_indices_val.csv').to_numpy().flatten()
	test_days = pd.read_csv(f'{data_path}/day_indices_test.csv').to_numpy().flatten()

	val_labels = pd.read_csv(f'{data_path}/labels_val.csv').to_numpy().flatten()
	test_labels = pd.read_csv(f'{data_path}/labels_test.csv').to_numpy().flatten()

	val_data = (val_labels,val_pids,val_days)
	test_data = (test_labels,test_pids,test_days)

	return val_data, test_data

def get_samples(args, dataset, model):
	
	
	batches = []
	data = []
	with DataLoader(dataset, model.config['num_first'], is_val=True, batch_size=1, seed=args.seed, device=args.device) as loader:
		for j, b in enumerate(loader):
			data.append(b)
			# if j == 0:
			# 	print(b['rnn'][0])
			# 	print(b['rnn'][1])
			# 	print(b['rnn'][2])
			# 	print(b['rnn'][3])
			# 	print(b['rnn'][4])
			# 	print(b['rnn'][5])
			# 	print(b['rnn'][6])
			# 	print(adsda)
	batch_idxs = np.random.choice([i for i in range(len(data))], args.batch_size, replace=False)
	# batch_ids, batch_days, batch_labels = [], [], []
	batch = {'pid':[],'day_index':[],'rnn':([],[],[],[],[],[],[]),'task':[],'label':[]}
	for idx in batch_idxs:
		batch['pid'].append(data[idx]['pid'][0])
		batch['day_index'].append(data[idx]['day_index'][0])
		batch['rnn'][0].append(data[idx]['rnn'][0])
		batch['rnn'][1].append(data[idx]['rnn'][1])
		batch['rnn'][2].append(data[idx]['rnn'][2])
		batch['rnn'][3].append(data[idx]['rnn'][3])
		batch['rnn'][4].append(data[idx]['rnn'][4])
		batch['rnn'][5].append(data[idx]['rnn'][5])
		batch['rnn'][6].append(data[idx]['rnn'][6])
		batch['task'].append(data[idx]['task'][0])
		batch['label'].append(data[idx]['label'][0][0])
		
		batch['rnn'][0] = torch.Tensor(batch['rnn'][0])
		batch['rnn'][1] = torch.Tensor(batch['rnn'][1])
		batch['rnn'][2] = torch.Tensor(batch['rnn'][2])
		batch['rnn'][3] = torch.Tensor(batch['rnn'][3])
		batch['rnn'][4] = torch.Tensor(batch['rnn'][4])
		batch['rnn'][5] = torch.Tensor(batch['rnn'][5])
		batch['rnn'][6] = torch.Tensor(batch['rnn'][6])

		batches.append(batch)
	return batches

def eval_alignment(args, model, dataset, clmbr_hp):
	align_list = []
	for i in range(args.iterations):
		batches = get_samples(args, dataset, model)
		# with DataLoader(dataset, model.config['num_first'], is_val=True, batch_size=99999, seed=args.seed, device=args.device) as loader:
		for j, batch in enumerate(batches):
			print(batch['rnn'])
			outputs = model(batch)

			z1 =outputs['z1']
			z2 = outputs['z2']
			align = alignment(z1, z2)

			align_list.append(align.item())
	return np.array(align_list)

def eval_uniform(args, model, dataset, clmbr_hp):
	uniform_list = []
	for i in range(args.iterations):
		dataset = get_sample(args, data, clmbr_hp)
		with DataLoader(dataset, model.config['num_first'], is_val=True, batch_size=99999, seed=args.seed, device=args.device) as loader:
			for batch in loader:
				outputs = model(batch)
				z1 = outputs['z1']
				uniform = uniformity(z1)
				# print(uniform)
				uniform_list.append(uniform.item())
	return np.array(uniform_list)

def alignment(x, y, alpha=2):
	return (x - y).norm(p=2, dim=1).pow(alpha).mean()

def uniformity(x, t=2):
	return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

def compute_ci(vals, metric,  ci=0.95):
	
	vals.sort()
	low_idx = int((1-ci)/2 * len(vals))-1

	up_idx = int((1+ci)/2 * len(vals))-1
	lower = vals[low_idx]
	upper = vals[up_idx]
	if len(vals) % 2 == 0:
		med = (vals[int(len(vals)/2-1)] + vals[int(len(vals)/2)])/2
	else:
		med = vals[int(math.floor(len(vals)/2))]
	df = pd.DataFrame({'metric':metric, 'lower_ci':lower, 'med_ci':med, 'upper_ci':upper}, index=[0])
	
	return df

def eval_model(args, val_dataset, test_dataset, clmbr_hp, cl_hp=None, pooler='BL'):
	
	if cl_hp:
		if pooler == 'CL-REP':
			clmbr_model_path = f'{args.clmbr_path}/cl_ete/models/best'
		elif pooler == 'OCP':
			clmbr_model_path = f'{args.clmbr_path}/ocp/models/best'
		else:
			clmbr_model_path = f'{args.clmbr_path}/contrastive_learn/models/{args.encoder}_sz_{clmbr_hp["size"]}_do_{clmbr_hp["dropout"]}_cd_{clmbr_hp["code_dropout"]}_dd_{clmbr_hp["day_dropout"]}_lr_{clmbr_hp["lr"]}_l2_{clmbr_hp["l2"]}/best'
	else:
		clmbr_model_path = f'{args.clmbr_path}/pretrained/models/{args.encoder}_sz_{clmbr_hp["size"]}_do_{clmbr_hp["dropout"]}_cd_{clmbr_hp["code_dropout"]}_dd_{clmbr_hp["day_dropout"]}_lr_{clmbr_hp["lr"]}_l2_{clmbr_hp["l2"]}'
	
	
	if cl_hp:
		if pooler == 'CL-REP':
			results_save_path = f'{args.results_path}/cl_rep/best'
		elif pooler == 'OCP':
			results_save_path = f'{args.results_path}/ocp/best'
		else:	
			results_save_path = f'{args.results_path}/contrastive_learn/{args.encoder}_sz_{clmbr_hp["size"]}_do_{clmbr_hp["dropout"]}_cd_{clmbr_hp["code_dropout"]}_dd_{clmbr_hp["day_dropout"]}_lr_{clmbr_hp["lr"]}_l2_{clmbr_hp["l2"]}/bs_{cl_hp["batch_size"]}_lr_{clmbr_hp["lr"]}_temp_{cl_hp["temp"]}_pool_{cl_hp["pool"]}'
	else:
		results_save_path = f'{args.results_path}/pretrained/{args.encoder}_sz_{clmbr_hp["size"]}_do_{clmbr_hp["dropout"]}_cd_{clmbr_hp["code_dropout"]}_dd_{clmbr_hp["day_dropout"]}_lr_{clmbr_hp["lr"]}_l2_{clmbr_hp["l2"]}'
	os.makedirs(f'{results_save_path}',exist_ok=True)
	#define model save path	
	clmbr_model = ehr_ml.clmbr.CLMBR.from_pretrained(clmbr_model_path, args.device).to(args.device)
	# set model to train so dropout is activated
	clmbr_model.train()
	config = clmbr_model.config
	
	dataset = PatientTimelineDataset(args.extract_path + '/extract.db', 
								 args.extract_path + '/ontology.db', 
								 f'{clmbr_model_path}/info.json', 
								 val_data, 
								 test_data)
	
	val_data_list = []
	for i in dataset.get_iterator(False, len(val_data), args.seed, config['num_first'], 0, 0):
		val_data_list.append(i)
	
	test_data_list = []
	for i in dataset.get_iterator(True, len(val_data), args.seed, config['num_first'], 0, 0):
		test_data_list.append(i)
	
	print(len(val_data_list))
	print(len(test_data_list))
	# test_dataset = PatientTimelineDataset(args.extract_path + '/extract.db', 
	# 							 args.extract_path + '/ontology.db', 
	# 							 f'{clmbr_model_path}/info.json', 
	# 							 test_data, 
	# 							 test_data)
	
	mw = MetricWrapper(clmbr_model, args.pooler, args.device)
	
	val_align= eval_alignment(args, mw, val_dataset, clmbr_hp)
	pd.DataFrame(val_align).reset_index(drop=True).to_csv(f'{results_save_path}/val_align_vals.csv')
	test_align = eval_alignment(args, mw, test_dataset, clmbr_hp)
	pd.DataFrame(test_align).reset_index(drop=True).to_csv(f'{results_save_path}/test_align_vals.csv')
	clmbr_model.eval()
	val_uniform = eval_uniform(args, mw, val_dataset, clmbr_hp)
	pd.DataFrame(val_uniform).reset_index(drop=True).to_csv(f'{results_save_path}/val_uniform_vals.csv')
	test_uniform = eval_uniform(args, mw, test_dataset, clmbr_hp)
	pd.DataFrame(test_uniform).reset_index(drop=True).to_csv(f'{results_save_path}/test_uniform_vals.csv')
	
	ci_df = pd.DataFrame()
	
	df = compute_ci(val_align, 'alignment')
	df['split'] = 'val'
	df['CLMBR'] = pooler
	ci_df = pd.concat([ci_df,df])
	
	df = compute_ci(val_uniform, 'uniformity')
	df['split'] = 'val'
	df['CLMBR'] = pooler
	ci_df = pd.concat([ci_df,df])
	
	df = compute_ci(test_align, 'alignment')
	df['split'] = 'test'
	df['CLMBR'] = pooler
	ci_df = pd.concat([ci_df,df])
	
	df = compute_ci(test_uniform, 'uniformity')
	df['split'] = 'test'
	df['CLMBR'] = pooler
	ci_df = pd.concat([ci_df,df])
	print(ci_df)
	ci_df.reset_index(drop=True).to_csv(f'{results_save_path}/uni_align_eval.csv')

#-------------------------------------------------------------------
# run
#-------------------------------------------------------------------
args = parser.parse_args()

# threads
joblib.Parallel(n_jobs=args.n_jobs)

# set seed
torch.manual_seed(args.seed)

grid = list(
	ParameterGrid(
		yaml.load(
			open(
				f"{os.path.join(args.hparams_fpath,args.encoder)}-do-best.yml",
				'r'
			),
			Loader=yaml.FullLoader
		)
	)
)


rd_grid = list(
    ParameterGrid(
        yaml.load(
            open(
                f"{os.path.join('/local-scratch/nigam/projects/jlemmon/cl-clmbr/experiments/main/artifacts/models/clmbr/contrastive_learn/models/gru_sz_800_do_0.1_cd_0_dd_0_lr_0.001_l2_0.01/best_rand_day','hyperparams')}.yml",
                'r'
            ),
            Loader=yaml.FullLoader
        )
    )
)

mr_grid = list(
    ParameterGrid(
        yaml.load(
            open(
                f"{os.path.join('/local-scratch/nigam/projects/jlemmon/cl-clmbr/experiments/main/artifacts/models/clmbr/contrastive_learn/models/gru_sz_800_do_0.1_cd_0_dd_0_lr_0.001_l2_0.01/best_mean_rep','hyperparams')}.yml",
                'r'
            ),
            Loader=yaml.FullLoader
        )
    )
)

dp_grid = list(
    ParameterGrid(
        yaml.load(
            open(
                f"{os.path.join('/local-scratch/nigam/projects/jlemmon/cl-clmbr/experiments/main/artifacts/models/clmbr/contrastive_learn/models/gru_sz_800_do_0.1_cd_0_dd_0_lr_0.001_l2_0.01/best_diff_pat','hyperparams')}.yml",
                'r'
            ),
            Loader=yaml.FullLoader
        )
    )
)

cl_rep_grid = list(
    ParameterGrid(
        yaml.load(
            open(
                f"{os.path.join('/local-scratch/nigam/projects/jlemmon/cl-clmbr/experiments/main/artifacts/models/clmbr/cl_ete/models/best','hyperparams')}.yml",
                'r'
            ),
            Loader=yaml.FullLoader
        )
    )
)

ocp_grid = list(
    ParameterGrid(
        yaml.load(
            open(
                f"{os.path.join('/local-scratch/nigam/projects/jlemmon/cl-clmbr/experiments/main/artifacts/models/clmbr/ocp/models/best','hyperparams')}.yml",
                'r'
            ),
            Loader=yaml.FullLoader
        )
    )
)

val_data, test_data = load_data(args, grid[0])



# print('BL')
# eval_model(args, val_data, test_data, grid[0])
# print('Random day CL')
# eval_model(args, val_data, test_data, grid[0], rd_grid[0], 'RD')
# print('Mean Representation CL')
# eval_model(args, val_data, test_data, grid[0], mr_grid[0], 'MR')
# print('Different Patient CL')
# eval_model(args, val_data, test_data, grid[0], dp_grid[0], 'DP')
# print('CL Representation')
# eval_model(args, val_data, test_data, grid[0], cl_rep_grid[0], 'CL-REP')
print('OCP Representation')
eval_model(args, val_data, test_data, grid[0], ocp_grid[0], 'OCP')
        
        
        

  