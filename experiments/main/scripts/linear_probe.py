import os
import json
import argparse
import shutil
import yaml
import copy
from datetime import datetime

import ehr_ml.timeline
import ehr_ml.ontology
import ehr_ml.index
import ehr_ml.labeler
import ehr_ml.clmbr
from ehr_ml.clmbr import Trainer
from ehr_ml.clmbr import PatientTimelineDataset
from ehr_ml.clmbr.dataset import DataLoader

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.model_selection import ParameterGrid
from sklearn.metrics import roc_auc_score
#from torch.utils.data import DataLoader, Dataset
from prediction_utils.pytorch_utils.metrics import StandardEvaluator


parser = argparse.ArgumentParser()

parser.add_argument(
    '--pt_model_path',
    type=str,
    default='/local-scratch/nigam/projects/jlemmon/cl-clmbr/experiments/main/artifacts/models/clmbr/pretrained/models',
    help='Base path for the pretrained model.'
)

parser.add_argument(
    '--model_path',
    type=str,
    default='/local-scratch/nigam/projects/jlemmon/cl-clmbr/experiments/main/artifacts/models/clmbr/contrastive_learn/models',
    help='Base path for the trained end-to-end model.'
)

parser.add_argument(
    '--ft_model_path',
    type=str,
    default='/local-scratch/nigam/projects/jlemmon/cl-clmbr/experiments/main/artifacts/models/clmbr/contrastive_learn/models/gru_sz_800_do_0.1_cd_0_dd_0_lr_0.001_l2_0.01/best',
    help='Base path for the best finetuned model.'
)

parser.add_argument(
    '--cl_rep_best_path',
    type=str,
    default='/local-scratch/nigam/projects/jlemmon/cl-clmbr/experiments/main/artifacts/models/clmbr/cl_ete/models/best',
    help='Base path for the best contrastively trained model.'
)

parser.add_argument(
    '--cl_rep_model_path',
    type=str,
    default='/local-scratch/nigam/projects/jlemmon/cl-clmbr/experiments/main/artifacts/models/clmbr/cl_ete/models',
    help='Base path for the trained CL rep model.'
)

parser.add_argument(
    '--ocp_best_path',
    type=str,
    default='/local-scratch/nigam/projects/jlemmon/cl-clmbr/experiments/main/artifacts/models/clmbr/ocp/models/best',
    help='Base path for the best contrastively trained OCP model.'
)

parser.add_argument(
    '--ocp_model_path',
    type=str,
    default='/local-scratch/nigam/projects/jlemmon/cl-clmbr/experiments/main/artifacts/models/clmbr/ocp/models',
    help='Base path for the trained OCP model.'
)

parser.add_argument(
    '--probe_path',
    type=str,
    default='/local-scratch/nigam/projects/jlemmon/cl-clmbr/experiments/main/artifacts/models/probes/baseline/models',
    help='Base path for the trained probe model.'
)

parser.add_argument(
    '--results_path',
    type=str,
    default='/local-scratch/nigam/projects/jlemmon/cl-clmbr/experiments/main/artifacts/results',
    help='Base path for the results.'
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
    '--cohort_dtype',
    type=str,
    default='parquet',
    help='Data type for cohort file.'
)

parser.add_argument(
    "--seed",
    type = int,
    default = 44,
    help = "seed for deterministic training"
)

parser.add_argument(
    '--batch_size',
    type=int,
    default=512,
    help='Size of training batch.'
)

parser.add_argument(
    '--epochs',
    type=int,
    default=20,
    help='Number of training epochs.'
)

parser.add_argument(
    '--n_boot',
    type=int,
    default=1000,
    help='Number of bootstrap iterations.'
)

parser.add_argument(
    '--n_jobs',
    type=int,
    default=8,
    help='Number of bootstrap jobs.'
)

parser.add_argument(
    '--early_stop',
    type=int,
    default=5,
    help='Number of training epochs before early stop is triggered.'
)

parser.add_argument(
    '--size',
    type=int,
    default=800,
    help='Size of embedding vector.'
)

parser.add_argument(
    '--dropout',
    type=float,
    default=0.1,
    help='Dropout proportion for training.'
)

parser.add_argument(
    '--pooler',
    type=str,
    default='cls',
    help='Pooler type to retrieve embedding.'
)

parser.add_argument(
    '--temp',
    type=float,
    default=0.05,
    help='Temperature value for the similarity scoring calculation'
)

parser.add_argument(
    '--lr',
    type=float,
    default=0.001,
    help='Learning rate for pretrained model.'
)

parser.add_argument(
	'--cl_lr',
	type=float,
	default=3e-5,
	help='Learning rate for constrastive learning finetuning.'
)

parser.add_argument(
    '--encoder',
    type=str,
    default='gru',
    help='Underlying neural network architecture for CLMBR. [gru|transformer|lstm]'
)

parser.add_argument(
    '--device',
    type=str,
    default='cuda:0',
    help='Device to run torch model on.'
)


class LinearProbe(nn.Module):
	def __init__(self, clmbr_model, size, device='cuda:0', is_ocp=False):
		super().__init__()
		self.clmbr_model = clmbr_model
		self.config = clmbr_model.config
		self.is_ocp = is_ocp
		
		self.dense = nn.Linear(size,1)
		self.activation = nn.Sigmoid()
		
		self.device = torch.device(device)
	
	def forward(self, batch):
		if self.is_ocp:
			features = self.clmbr_model.timeline_model(batch['rnn'])[0].to(self.device)
		else:
			features = self.clmbr_model.timeline_model(batch['rnn']).to(self.device)

		label_indices, label_values = batch['label']
		
		flat_features = features.view((-1, features.shape[-1]))
		target_features = F.embedding(label_indices, flat_features).to(args.device)
									
		preds = self.activation(self.dense(target_features))
		
		return preds, label_values.to(torch.float32)
	
	def freeze_clmbr(self):
		self.clmbr_model.freeze()
	
	def unfreeze_clmbr(self):
		self.clmbr_model.unfreeze()

	
def load_datasets(args, task, clmbr_hp, clmbr_model_path):
	"""
	Load datasets from split csv files.
	"""
	data_path = f'{args.labelled_fpath}/{task}/pretrained/{args.encoder}_sz_{clmbr_hp["size"]}_do_{clmbr_hp["dropout"]}_cd_{clmbr_hp["code_dropout"]}_dd_{clmbr_hp["day_dropout"]}_lr_{clmbr_hp["lr"]}_l2_{clmbr_hp["l2"]}'

	train_pids = pd.read_csv(f'{data_path}/ehr_ml_patient_ids_train.csv')
	val_pids = pd.read_csv(f'{data_path}/ehr_ml_patient_ids_val.csv')
	test_pids = pd.read_csv(f'{data_path}/ehr_ml_patient_ids_test.csv')

	train_days = pd.read_csv(f'{data_path}/day_indices_train.csv')
	val_days = pd.read_csv(f'{data_path}/day_indices_val.csv')
	test_days = pd.read_csv(f'{data_path}/day_indices_test.csv')

	train_labels = pd.read_csv(f'{data_path}/labels_train.csv')
	val_labels = pd.read_csv(f'{data_path}/labels_val.csv')
	test_labels = pd.read_csv(f'{data_path}/labels_test.csv')

	train_data = (train_labels.to_numpy().flatten(),train_pids.to_numpy().flatten(),train_days.to_numpy().flatten())
	val_data = (val_labels.to_numpy().flatten(),val_pids.to_numpy().flatten(),val_days.to_numpy().flatten())
	test_data = (test_labels.to_numpy().flatten(),test_pids.to_numpy().flatten(),test_days.to_numpy().flatten())
	
	train_dataset = PatientTimelineDataset(args.extract_path + '/extract.db', 
											 args.extract_path + '/ontology.db', 
											 f'{clmbr_model_path}/info.json', 
											 train_data, 
											 val_data )
	
	test_dataset = PatientTimelineDataset(args.extract_path + '/extract.db', 
										 args.extract_path + '/ontology.db', 
										 f'{clmbr_model_path}/info.json', 
										 train_data, 
										 test_data )
    
	return train_dataset, test_dataset


def train_probe(args, model, dataset, save_path):
	"""
	Train linear classification probe on frozen CLMBR model.
	At each epoch save model if validation loss was improved.
	"""
	model.train()
	model.freeze_clmbr()
	optimizer = optim.Adam([p for n, p in model.named_parameters() if p.requires_grad], lr=args.lr)
	
	criterion = nn.BCELoss()
	
	best_model = None
	best_val_loss = 9999999
	best_val_preds = []
	best_val_lbls = []
	best_val_ids = []
	
	for e in range(args.epochs):
		val_preds = []
		val_lbls = []
		val_ids = []
		print(f'epoch {e+1}/{args.epochs}')
		epoch_train_loss = 0.0
		epoch_val_loss = 0.0
		# Iterate through training data loader
		with DataLoader(dataset, model.config['num_first'], is_val=False, batch_size=model.config["batch_size"], device=args.device) as train_loader:
			for batch in train_loader:

				optimizer.zero_grad()
				logits, labels = model(batch)
				loss = criterion(logits, labels.unsqueeze(-1))

				loss.backward()
				optimizer.step()
				epoch_train_loss += loss.item()
				
		# Iterate through validation data loader
		with torch.no_grad():
			with DataLoader(dataset, 9262, is_val=True, batch_size=model.config["batch_size"], device=args.device) as val_loader:
				for batch in val_loader:
					logits, labels = model(batch)
					loss = criterion(logits, labels.unsqueeze(-1))
					epoch_val_loss += loss.item()
					val_preds.extend(logits.cpu().numpy().flatten())
					val_lbls.extend(labels.cpu().numpy().flatten())
					val_ids.extend(batch['pid'])
				# val_losses.append(epoch_val_loss)
		
		#print epoch losses
		print('epoch train loss:', epoch_train_loss)
		print('epoch val loss:', epoch_val_loss)
		
		# save model if validation loss is improved
		if epoch_val_loss < best_val_loss:
			print('Saving best model...')
			best_val_loss = epoch_val_loss
			best_model = copy.deepcopy(model)
			torch.save(best_model, f'{save_path}/best_model.pth')
			
			# flatten prediction and label arrays
			best_val_preds = val_preds
			best_val_lbls = val_lbls
			best_val_ids = val_ids
	return best_model, best_val_preds, best_val_lbls, best_val_ids

def evaluate_probe(args, model, dataset):
	model.eval()
	
	criterion = nn.BCELoss()
	
	preds = []
	lbls = []
	ids = []
	with torch.no_grad():
		with DataLoader(dataset, model.config['num_first'], is_val=True, batch_size=model.config['batch_size'], seed=args.seed, device=args.device) as eval_loader:
			for batch in eval_loader:
				if (len(batch['pid']) != len(batch['label'][0])):
					batch['pid'] = batch['pid'][:len(batch['label'][0])] #temp fix
				logits, labels = model(batch)
				loss = criterion(logits, labels.unsqueeze(-1))
				# losses.append(loss.item())
				preds.extend(logits.cpu().numpy().flatten())
				lbls.extend(labels.cpu().numpy().flatten())
				ids.extend(batch['pid'])
	return preds, lbls, ids
			
def calc_metrics(args, df):
	evaluator = StandardEvaluator()
	eval_ci_df, eval_df = evaluator.bootstrap_evaluate(
		df,
		n_boot = args.n_boot,
		n_jobs = args.n_jobs,
		strata_vars_eval = ['phase'],
		strata_vars_boot = ['labels'],
		patient_id_var='person_id',
		return_result_df = True
	)
	eval_ci_df['model'] = 'probe'
	return eval_ci_df
	
if __name__ == '__main__':
	args = parser.parse_args()
	
	torch.manual_seed(args.seed)
	tasks = ['hospital_mortality', 'LOS_7', 'icu_admission', 'readmission_30']#, 'sudden_cardiac_death', 'stroke', 'bladder_cancer', 'breast_cancer', 'acute_renal_failure', 'acute_myocardial_infarction', 'diabetic_ketoacidosis', 'edema', 'hyperkylemia', 'renal_cancer', 'revascularization']
	
	# load best CLMBR model parameter grid
	bl_hp = list(
		ParameterGrid(
			yaml.load(
				open(
					f"{os.path.join(args.hparams_fpath,args.encoder)}-do-best.yml",
					'r'
				),
				Loader=yaml.FullLoader
			)
		)
	)[0]
	
# 	rd_hp = list(
# 		ParameterGrid(
# 			yaml.load(
# 				open(
# 					f"{os.path.join(args.ft_model_path + '_' + 'rand_day','hyperparams')}.yml",
# 					'r'
# 				),
# 				Loader=yaml.FullLoader
# 			)
# 		)
# 	)[0]
	
# 	dp_hp = list(
# 		ParameterGrid(
# 			yaml.load(
# 				open(
# 					f"{os.path.join(args.ft_model_path + '_' + 'diff_pat','hyperparams')}.yml",
# 					'r'
# 				),
# 				Loader=yaml.FullLoader
# 			)
# 		)
# 	)[0]
	
	mr_hp = list(
		ParameterGrid(
			yaml.load(
				open(
					f"{os.path.join(args.ft_model_path,'hyperparams')}.yml",
					'r'
				),
				Loader=yaml.FullLoader
			)
		)
	)[0]
	
	cl_rep_hp = list(
		ParameterGrid(
			yaml.load(
				open(
					f"{os.path.join(args.cl_rep_best_path,'hyperparams')}.yml",
					'r'
				),
				Loader=yaml.FullLoader
			)
		)
	)[0]
	
	ocp_hp = list(
		ParameterGrid(
			yaml.load(
				open(
					f"{os.path.join(args.ocp_best_path,'hyperparams')}.yml",
					'r'
				),
				Loader=yaml.FullLoader
			)
		)
	)[0]
	
	# Iterate through tasks
	for task in tasks:
		print(f'Task {task}')
		
		# Iterate through (singular) CLMBR hyperparam settings
	
		print('Training BL CLMBR probe with params: ', bl_hp)
			
# 		# Path where CLMBR model is saved
		bl_model_str = f'{args.encoder}_sz_{bl_hp["size"]}_do_{bl_hp["dropout"]}_cd_{bl_hp["code_dropout"]}_dd_{bl_hp["day_dropout"]}_lr_{bl_hp["lr"]}_l2_{bl_hp["l2"]}'
		clmbr_model_path = f'{args.pt_model_path}/{bl_model_str}'
# 		print(clmbr_model_path)

# 		# Load  datasets
		train_dataset, test_dataset = load_datasets(args, task, bl_hp, clmbr_model_path)

# 		# Path where CLMBR probe will be saved
		probe_save_path = f'{args.probe_path}/{task}/baseline/{bl_model_str}'
		os.makedirs(f"{probe_save_path}",exist_ok=True)
			
		result_save_path = f'{args.results_path}/{task}/probes/baseline/{bl_model_str}'
		os.makedirs(f"{result_save_path}",exist_ok=True)
			
		# Load CLMBR model and attach linear probe
# 		clmbr_model = ehr_ml.clmbr.CLMBR.from_pretrained(clmbr_model_path, args.device).to(args.device)
# 		clmbr_model.freeze()

# 		probe_model = LinearProbe(clmbr_model, bl_hp['size'])

# 		probe_model.to(args.device)

# 		print('Training probe...')
# 		# Train probe and evaluate on validation 
# 		probe_model, val_preds, val_labels, val_ids = train_probe(args, probe_model, train_dataset, probe_save_path)

# 		val_df = pd.DataFrame({'CLMBR':'BL', 'model':'linear', 'task':task, 'phase':'val', 'person_id':val_ids, 'pred_probs':val_preds, 'labels':val_labels})
# 		val_df.to_csv(f'{result_save_path}/val_preds.csv',index=False)

# 		print('Testing probe...')
# 		test_preds, test_labels, test_ids = evaluate_probe(args, probe_model, test_dataset)

# 		test_df = pd.DataFrame({'CLMBR':'BL', 'model':'linear', 'task':task, 'phase':'test', 'person_id':test_ids, 'pred_probs':test_preds, 'labels':test_labels})
# 		test_df.to_csv(f'{result_save_path}/test_preds.csv',index=False)
# 		df_preds = pd.concat((val_df,test_df))
# 		df_preds['CLMBR'] = df_preds['CLMBR'].astype(str)
# 		df_preds['model'] = df_preds['model'].astype(str)
# 		df_preds['task'] = df_preds['task'].astype(str)
# 		df_preds['phase'] = df_preds['phase'].astype(str)

# 		df_eval = calc_metrics(args, df_preds)
# 		df_eval['CLMBR'] = 'BL'
# 		df_eval['task'] = task
# 		df_eval.to_csv(f'{result_save_path}/eval.csv',index=False)
		
		for cl_hp in [mr_hp]:#[dp_hp, rd_hp, mr_hp]:
			print('Training CL-CLMBR probe with params: ', cl_hp)
			cl_model_str = f'bs_{cl_hp["batch_size"]}_lr_{cl_hp["lr"]}_temp_{cl_hp["temp"]}_pool_{cl_hp["pool"]}'
			cl_model_path = f"{args.ft_model_path}_{cl_hp['pool']}"
			# Create probe and result directories
			probe_save_path = f'{args.probe_path}/{task}/contrastive_learn/{bl_model_str}/{cl_model_str}'

			os.makedirs(f"{probe_save_path}",exist_ok=True)

			result_save_path = f'{args.results_path}/{task}/probes/contrastive_learn/{bl_model_str}/{cl_model_str}'

			os.makedirs(f"{result_save_path}",exist_ok=True)

			# load cl clmbr model
			cl_model = ehr_ml.clmbr.CLMBR.from_pretrained(clmbr_model_path, args.device).to(args.device)
			cl_model.freeze()

			# Get probe model
			cl_probe_model = LinearProbe(cl_model, bl_hp['size'])

			cl_probe_model.to(args.device)

			# Train probe and get best model by validation score
			cl_probe_model, val_preds, val_labels, val_ids = train_probe(args, cl_probe_model, train_dataset,  probe_save_path)
			cl_str = 'CL-'
			if cl_hp['pool'] == 'mean_rep':
				cl_str += 'MR'
			elif cl_hp['pool'] == 'rand_day':
				cl_str += 'RD'
			elif cl_hp['pool'] == 'diff_pat':
				cl_str += 'DP'
			val_df = pd.DataFrame({'CLMBR':cl_str, 'model':'linear','task':task, 'phase':'val', 'person_id':val_ids, 'pred_probs':val_preds, 'labels':val_labels})
			val_df.to_csv(f'{result_save_path}/val_preds.csv',index=False)

			# Run probe on test set
			print('Testing probe...')

			test_preds, test_labels, test_ids = evaluate_probe(args, cl_probe_model, test_dataset)
			test_df = pd.DataFrame({'CLMBR':cl_str, 'model':'linear', 'task':task, 'phase':'test', 'person_id':test_ids, 'pred_probs':test_preds, 'labels':test_labels})
			test_df.to_csv(f'{result_save_path}/test_preds.csv',index=False)

			# create pred prob df and bootstrap metrics
			df_preds = pd.concat((val_df,test_df))
			df_eval = calc_metrics(args, df_preds)
			df_eval['CLMBR'] = cl_str
			df_eval['task'] = task
			print(df_eval)
			df_eval.to_csv(f'{result_save_path}/eval.csv',index=False)

		
		print('Training CL Representation probe with params: ', cl_rep_hp)
			
		# Path where CLMBR model is saved
		cl_model_str = f'gru_sz_800_do_0.1_cd_0_dd_0_lr_0.01_l2_0.1_bs_2000_lr_5e-5_temp_0.01_pool_mean_rep'
		clmbr_model_path = f'{args.cl_rep_best_path}'
		print(clmbr_model_path)


		# Path where CLMBR probe will be saved
		probe_save_path = f'{args.probe_path}/{task}/cl_rep/{cl_model_str}'
		os.makedirs(f"{probe_save_path}",exist_ok=True)
			
		result_save_path = f'{args.results_path}/{task}/probes/cl_rep/{cl_model_str}'
		os.makedirs(f"{result_save_path}",exist_ok=True)
			
		# Load CLMBR model and attach linear probe
		clmbr_model = ehr_ml.clmbr.CLMBR.from_pretrained(clmbr_model_path, args.device).to(args.device)
		clmbr_model.freeze()

		probe_model = LinearProbe(clmbr_model, bl_hp['size'])

		probe_model.to(args.device)

		print('Training probe...')
		# Train probe and evaluate on validation 
		probe_model, val_preds, val_labels, val_ids = train_probe(args, probe_model, train_dataset, probe_save_path)

		val_df = pd.DataFrame({'CLMBR':'CL_REP', 'model':'linear', 'task':task, 'phase':'val', 'person_id':val_ids, 'pred_probs':val_preds, 'labels':val_labels})
		val_df.to_csv(f'{result_save_path}/val_preds.csv',index=False)

		print('Testing probe...')
		test_preds, test_labels, test_ids = evaluate_probe(args, probe_model, test_dataset)

		test_df = pd.DataFrame({'CLMBR':'CL_REP', 'model':'linear', 'task':task, 'phase':'test', 'person_id':test_ids, 'pred_probs':test_preds, 'labels':test_labels})
		test_df.to_csv(f'{result_save_path}/test_preds.csv',index=False)
		df_preds = pd.concat((val_df,test_df))
		df_preds['CLMBR'] = df_preds['CLMBR'].astype(str)
		df_preds['model'] = df_preds['model'].astype(str)
		df_preds['task'] = df_preds['task'].astype(str)
		df_preds['phase'] = df_preds['phase'].astype(str)

		df_eval = calc_metrics(args, df_preds)
		df_eval['CLMBR'] = 'CL_REP'
		df_eval['task'] = task
		df_eval.to_csv(f'{result_save_path}/eval.csv',index=False)
		
		print('Training OCP probe with params: ', ocp_hp)
			
		# Path where CLMBR model is saved
		cl_model_str = f'gru_sz_800_do_0.2_l2_0.01_lr_5e-5_pool_ocp'
		clmbr_model_path = f'{args.ocp_best_path}'
		print(clmbr_model_path)


		# Path where CLMBR probe will be saved
		probe_save_path = f'{args.probe_path}/{task}/ocp/{cl_model_str}'
		os.makedirs(f"{probe_save_path}",exist_ok=True)
			
		result_save_path = f'{args.results_path}/{task}/probes/ocp/{cl_model_str}'
		os.makedirs(f"{result_save_path}",exist_ok=True)
			
		# Load CLMBR model and attach linear probe
		clmbr_model = ehr_ml.clmbr.CLMBR.from_pretrained(clmbr_model_path, args.device).to(args.device)
		clmbr_model.freeze()

		probe_model = LinearProbe(clmbr_model, bl_hp['size'], is_ocp=True)

		probe_model.to(args.device)

		print('Training probe...')
		# Train probe and evaluate on validation 
		probe_model, val_preds, val_labels, val_ids = train_probe(args, probe_model, train_dataset, probe_save_path)

		val_df = pd.DataFrame({'CLMBR':'OCP', 'model':'linear', 'task':task, 'phase':'val', 'person_id':val_ids, 'pred_probs':val_preds, 'labels':val_labels})
		val_df.to_csv(f'{result_save_path}/val_preds.csv',index=False)

		print('Testing probe...')
		test_preds, test_labels, test_ids = evaluate_probe(args, probe_model, test_dataset)

		test_df = pd.DataFrame({'CLMBR':'OCP', 'model':'linear', 'task':task, 'phase':'test', 'person_id':test_ids, 'pred_probs':test_preds, 'labels':test_labels})
		test_df.to_csv(f'{result_save_path}/test_preds.csv',index=False)
		df_preds = pd.concat((val_df,test_df))
		df_preds['CLMBR'] = df_preds['CLMBR'].astype(str)
		df_preds['model'] = df_preds['model'].astype(str)
		df_preds['task'] = df_preds['task'].astype(str)
		df_preds['phase'] = df_preds['phase'].astype(str)

		df_eval = calc_metrics(args, df_preds)
		df_eval['CLMBR'] = 'OCP'
		df_eval['task'] = task
		df_eval.to_csv(f'{result_save_path}/eval.csv',index=False)
