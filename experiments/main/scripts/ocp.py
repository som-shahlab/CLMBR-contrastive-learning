import os
import json
import argparse
import shutil
import yaml
import random
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
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('./runs/cl_ete')

from sklearn.model_selection import ParameterGrid
#from torch.utils.data import DataLoader, Dataset


parser = argparse.ArgumentParser()

parser.add_argument(
    '--pt_info_path',
    type=str,
    default='/local-scratch/nigam/projects/jlemmon/cl-clmbr/experiments/main/artifacts/models/clmbr/pretrained/info',
    help='Base path for the pretrained model info.'
)

parser.add_argument(
    '--model_path',
    type=str,
    default='/local-scratch/nigam/projects/jlemmon/cl-clmbr/experiments/main/artifacts/models/clmbr/ocp/models',
    help='Base path for the trained end-to-end model.'
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
    default="/local-scratch/nigam/projects/jlemmon/cl-clmbr/experiments/main/data/labelled_data/readmission_30/pretrained/gru_sz_800_do_0.1_cd_0_dd_0_lr_0.001_l2_0.01",
    help='Base path for labelled data directory'
)

parser.add_argument(
    '--filtered_id_fpath',
    type=str,
    default="/local-scratch/nigam/projects/jlemmon/cl-clmbr/experiments/main/data/ocp_patients",
    help='Base path for patient ids with >2 admissions data directory'
)

parser.add_argument(
    '--ocp_pid_fpath',
    type=str,
    default="/local-scratch/nigam/projects/jlemmon/cl-clmbr/experiments/main/data/ocp_patients",
    help='Base path for converted OCP patient ids directory'
)


parser.add_argument(
    '--train_end_date',
    type=str,
    default='2016-12-31',
    help='End date of training ids.'
)

parser.add_argument(
    '--val_end_date',
    type=str,
    default='2016-12-31',
    help='End date of validation ids.'
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
    default=1,
    help='Size of training batch.'
)

parser.add_argument(
    '--epochs',
    type=int,
    default=100,
    help='Number of training epochs.'
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
    default='ocp',
    help='Pooler type to retrieve embedding.'
)

parser.add_argument(
    '--lr',
    type=float,
    default=0.001,
    help='Learning rate for pretrained model.'
)

parser.add_argument(
    '--encoder',
    type=str,
    default='gru',
    help='Underlying neural network architecture for CLMBR. [gru|transformer|lstm]'
)

parser.add_argument(
    '--l2',
    type=float,
    default=0.1,
    help='L2 regularization value.'
)

parser.add_argument(
    '--device',
    type=str,
    default='cuda:0',
    help='Device to run torch model on.'
)

parser.add_argument(
    '--warmup_epochs',
    type=int,
    default=2,
    help='Size of embedding vector.'
)

parser.add_argument(
    '--multi_gpu',
    type=int,
    default=0,
    help='If model gridsearch is split across multiple GPUs or not.'
)

parser.add_argument(
	'--patience',
	type=int,
	default=10,
	help='Number of epochs to wait before triggering early stopping.'
)

parser.add_argument(
    '--idx_end_date',
    type=str,
    default='2016-12-31',
    help='End date of training/validation ids.'
)

class MLPLayer(nn.Module):
	"""
	Linear classifier layer.
	"""
	def __init__(self, size):
		super().__init__()
		self.dense = nn.Linear(size, 1)
		self.activation = nn.Sigmoid()
		
	def forward(self, features):
		x = self.dense(features)
		x = self.activation(x)
		
		return x

class Pooler(nn.Module):
	"""
	Pooler module to get the pooled embedding value. 
	"""
	def __init__(self, pooler='ocp', device=None):
		super().__init__()
		self.pooler = pooler
		self.device = device if device is not None else 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
	def forward(self, embeds, day_indices=None):
		if self.pooler == 'ocp':
			return embeds
		else:
			return embeds[-1]

class ContrastiveLearn(nn.Module):
	"""
	Linear layer for end-to-end finetuning and prediction.
	"""
	def __init__(self, clmbr_model, num_classes, pooler, temp, device=None):
		super().__init__()
		self.clmbr_model = clmbr_model
		self.config = clmbr_model.config
		self.linear = MLPLayer(clmbr_model.config["size"])
		self.pooler = Pooler(pooler, device)
		self.criterion = nn.BCELoss()
		self.device = device if device is not None else 'cuda:0' if torch.cuda.is_available() else 'cpu'

	def forward(self, batch, windows, is_train=True):
		outputs = dict()
		
		# For patient timeline in batch get window pair embeddings
		embeds, labels = self.clmbr_model.timeline_model(batch["rnn"], windows)
		labels = labels.to(self.device)
		
		# Use pooler to get target embeddings
		target_embeds = self.pooler(embeds)
		
		preds = self.linear(target_embeds)
		preds = preds.squeeze(-1)
		preds = torch.mean(preds, dim=1)

# 		# Compute loss
		outputs['loss'] = self.criterion(preds,labels)
		outputs['preds'] = preds
		outputs['labels'] = labels
		return outputs

class EarlyStopping():
	def __init__(self, patience):
		self.patience = patience
		self.early_stop = False
		self.best_loss = 9999999
		self.counter = 0
	def __call__(self, loss):
		if self.best_loss > loss:
			self.counter = 0
			self.best_loss = loss
		else:
			self.counter += 1
			if self.counter == self.patience:
				self.early_stop = True
		return self.early_stop		
	
def load_data(args):
	"""
	Load datasets from split csv files.
	"""
	data_path = f'{args.labelled_fpath}'
	filtered_id_path = f'{args.filtered_id_fpath}'
	
	#get list of patient ids with >1 admission
	admission_df = pd.read_csv(f'{filtered_id_path}/patients.csv')
	admission_df = admission_df.drop(admission_df[admission_df.admit_date > args.idx_end_date].index)
	pids = np.array(admission_df['ehr_id'].unique())
	
	admission_df = get_adm_window_indices(admission_df, pids)
	
	# get indices of pids with >1 admission from data splits
	train_pids = pd.read_csv(f'{data_path}/ehr_ml_patient_ids_train.csv')
	train_idx = train_pids.index[train_pids['0'].isin(pids)]
	
	val_pids = pd.read_csv(f'{data_path}/ehr_ml_patient_ids_val.csv')
	val_idx = val_pids.index[val_pids['0'].isin(pids)]
	
	train_days = pd.read_csv(f'{data_path}/day_indices_train.csv')
	val_days = pd.read_csv(f'{data_path}/day_indices_val.csv')

	train_labels = pd.read_csv(f'{data_path}/labels_train.csv')
	val_labels = pd.read_csv(f'{data_path}/labels_val.csv')
	
	w_pid_tr = train_pids.iloc[train_idx]['0']
	w_pid_vl = val_pids.iloc[val_idx]['0']
	
	train_adm_days = admission_df.query('ehr_id.isin(@w_pid_tr)')[['ehr_id','end_day_idx']].groupby('ehr_id').max().reindex(train_pids.to_numpy().flatten()[train_idx])
	val_adm_days = admission_df.query('ehr_id.isin(@w_pid_vl)')[['ehr_id','end_day_idx']].groupby('ehr_id').max().reindex(val_pids.to_numpy().flatten()[val_idx])

	train_data = (train_labels.to_numpy().flatten()[train_idx],train_pids.to_numpy().flatten()[train_idx],train_adm_days.to_numpy().flatten())
	val_data = (val_labels.to_numpy().flatten()[val_idx],val_pids.to_numpy().flatten()[val_idx],val_adm_days.to_numpy().flatten())

	return train_data, val_data, admission_df
    
def get_adm_window_indices(adm_df, pids):
	# get indices of start and end of admission windows
	adm_df['diff'] = ((pd.to_datetime(adm_df['discharge_date']).dt.date - pd.to_datetime(adm_df['admit_date']).dt.date) / np.timedelta64(1,'D')).astype('int')
	adm_df['cum_days'] = adm_df.groupby(['ehr_id'])['diff'].cumsum().astype('int')
	adm_df['win_idx_start'] = adm_df['cum_days'] - adm_df['diff']
	adm_df['win_idx_end'] = adm_df['cum_days']
	# adm_df = adm_df.drop(columns=['cum_days', 'diff'])

	return adm_df
	
def train(args, model, train_dataset, windows, lr, clmbr_save_path, clmbr_info_path, hp_int):
	"""
	Train CLMBR model using CL objective.
	"""
	model.train()
	# config = model.clmbr_model.config
	config = model.config
	optimizer = optim.Adam([p for n, p in model.named_parameters() if p.requires_grad], lr=lr)
	best_val_loss = 9999999
	early_stop = EarlyStopping(args.patience)
	train_output_df = pd.DataFrame()
	val_output_df = pd.DataFrame()
	model_train_loss = []
	model_val_loss = []
	best_epoch = 0
	for e in range(args.epochs):
		model.train()
		train_loss = []
		train_preds = []
		train_lbls = []
		with DataLoader(train_dataset, model.config['num_first'], is_val=False, batch_size=1, seed=args.seed, device=args.device) as train_loader:
			for batch in train_loader:
					 
				pid_df = windows.query('ehr_id==@batch["pid"][0]')
				if len(pid_df) < 2:
					continue
				else:
					#drop last admission if odd number
					if len(pid_df) % 2 != 0:
						pid_df = pid_df[:-1]
					optimizer.zero_grad()
					outputs = model(batch, pid_df)
					train_preds.extend(list(outputs['preds'].detach().clone().cpu().numpy()))
					train_lbls.extend(list(outputs['labels'].detach().clone().cpu().numpy()))
					loss = outputs["loss"]
					loss.backward()
					optimizer.step()
					train_loss.append(loss.item())

		print('Training loss:',  np.sum(train_loss))
		model_train_loss.append(np.sum(train_loss))
		
		# evaluate on validation set
		val_preds, val_lbls, val_losses = evaluate_model(args, model, val_data, windows)
		val_loss = np.sum(val_losses)
		model_val_loss.append(val_loss)
		
		
		# Save train and val model predictions/labels
		df = pd.DataFrame({'epoch':e,'preds':train_preds,'labels':train_lbls})
		train_output_df = pd.concat((train_output_df,df),axis=0)
		df = pd.DataFrame({'epoch':e,'preds':val_preds,'labels':val_lbls})
		val_output_df = pd.concat((val_output_df,df),axis=0)

		#save current epoch model
		os.makedirs(f'{clmbr_save_path}/{e}',exist_ok=True)
		torch.save(clmbr_model.state_dict(), os.path.join(clmbr_save_path,f'{e}/best'))
		shutil.copyfile(f'{clmbr_info_path}', f'{clmbr_save_path}/{e}/info.json')
		with open(f'{clmbr_save_path}/{e}/config.json', 'w') as f:
			json.dump(config,f)
			
		# save model as best model if condition met
		if val_loss < best_val_loss:
			best_val_loss = val_loss
			best_epoch = e
			best_model = copy.deepcopy(model.clmbr_model)
			
		print('Epoch train loss', np.sum(train_loss))
		print('Epoch val loss', val_loss)
		# Trigger early stopping if model hasn't improved for awhile
		if early_stop(scaled_val_loss):
			print(f'Early stopping at epoch {e}')
			break
		
	# write train and val loss/preds to csv
	df = pd.DataFrame(model_train_loss)
	df.to_csv(f'{clmbr_save_path}/train_loss.csv')
	df = pd.DataFrame(model_val_loss)
	df.to_csv(f'{clmbr_save_path}/val_loss.csv')
	train_output_df.to_csv(f'{clmbr_save_path}/train_preds.csv', index=False)
	val_output_df.to_csv(f'{clmbr_save_path}/val_preds.csv', index=False)
	with open(f'{clmbr_save_path}/best_epoch.txt', 'w') as f:
		f.write(f'{best_epoch}')
		
	return best_model, best_val_loss, val_output_df

def evaluate_model(args, model, data, windows):
	model.eval()

	criterion = nn.CrossEntropyLoss()

	preds = []
	lbls = []
	losses = []
	with torch.no_grad():
			# print(pid_df)
			dataset = PatientTimelineDataset(args.extract_path + '/extract.db', 
											 args.extract_path + '/ontology.db', 
											 f'{clmbr_info_path}', 
											 data, 
											 data )
			with DataLoader(dataset, model.config['num_first'], is_val=True, batch_size=1, seed=args.seed, device=args.device) as eval_loader:
				for batch in eval_loader:
					
					pid_df = windows.query('ehr_id==@batch["pid"][0]')
					if len(pid_df) < 2:
						pass
					else:
						if len(pid_df) % 2 != 0:
							pid_df = pid_df[:-1]
						outputs = model(batch, pid_df)
						loss = outputs["loss"]
						losses.append(loss.item())
						preds.extend(list(outputs['preds'].cpu().numpy()))
						lbls.extend(list(outputs['labels'].cpu().numpy()))
	print('Validation loss:',  np.sum(losses))
	return preds, lbls, losses

def get_config(hp):
    config = {'b1': 0.9,
            'b2': 0.999,
            'batch_size': 1000,
            'code_dropout': 0,
            'day_dropout': 0,
            'dropout': hp['dropout'],
            'e': 1e-08,
            'encoder_type': hp['encoder_type'],
            'epochs_per_cycle': 1,
            'eval_batch_size': 1000,
            'l2': hp['l2'],
            'lr': hp['lr'],
            'model_dir': '',
            'num_first': 9262,
            'num_second': 10044,
            'rnn_layers': 1,
            'size': hp['size'],
            'tied_weights': True,
			'ocp':1,
            'warmup_epochs': hp['warmup_epochs']}
    return config

if __name__ == '__main__':
    
	args = parser.parse_args()

	torch.manual_seed(args.seed)
	
	if args.multi_gpu == 1:
		grid = list(
			[{'lr':args.lr,
			 'pool':args.pooler,
			 'encoder_type':args.encoder,
			 'size':args.size,
			 'dropout':args.dropout,
			 'epochs':args.epochs,
			 'warmup_epochs':args.warmup_epochs,
			 'l2':args.l2}]
		)
	else:
		grid = list(
			ParameterGrid(
				yaml.load(
					open(
						f"{os.path.join(args.hparams_fpath,'ocp')}.yml",
						'r'
					),
					Loader=yaml.FullLoader
				)
			)
		)
	for i, hp in enumerate(grid):
		print('Initialized OCP model with params: ', hp)
		clmbr_info_path = f'{args.pt_info_path}/info.json'
		with open(clmbr_info_path) as f:
			info = json.load(f)
		config = get_config(hp)
		best_val_loss = 9999999
		best_params = None

		model_str = f"{args.encoder}_sz_{hp['size']}_do_{hp['dropout']}_l2_{hp['l2']}_lr_{hp['lr']}_pool_{hp['pool']}"
		clmbr_save_path = f"{args.model_path}/{model_str}"
		print(clmbr_save_path)
		os.makedirs(f"{clmbr_save_path}",exist_ok=True)
		train_data, val_data, windows = load_data(args)
		train_dataset = PatientTimelineDataset(args.extract_path + '/extract.db', 
										 args.extract_path + '/ontology.db', 
										 f'{clmbr_info_path}', 
										 train_data, 
										 val_data )
		
		config["model_dir"] = clmbr_save_path
		config['lr'] = hp['lr']
		clmbr_model = ehr_ml.clmbr.CLMBR(config, info).to(torch.device(args.device))

		config = clmbr_model.config

		clmbr_model.unfreeze()
		# Get contrastive learning model 
		model = ContrastiveLearn(clmbr_model, 2, 'ocp', args.device).to(args.device)
		model.train()

		# Run finetune procedure
		clmbr_model, val_loss, val_df = train(args, model, train_dataset, windows, float(hp['lr']), clmbr_save_path, clmbr_info_path, i)
		writer.flush()
		clmbr_model.freeze()
		if val_loss < best_val_loss:
			print('Saving as best trained model...')
			best_val_loss = val_loss
			best_params = hp
			best_path = os.path.join(args.model_path,'best')
			os.makedirs(f"{best_path}",exist_ok=True)

			torch.save(clmbr_model.state_dict(), f'{best_path}/best')
			shutil.copyfile(clmbr_info_path, f'{best_path}/info.json')
			with open(f'{best_path}/config.json', 'w') as f:
				json.dump(config,f)
			with open(f"{best_path}/hyperparams.yml", 'w') as file: # fix format of dump
				yaml.dump(best_params,file)
			val_df.to_csv(f'{best_path}/val_preds.csv', index=False)
		# Save model and save info and config to new model directory for downstream evaluation
		torch.save(clmbr_model.state_dict(), os.path.join(clmbr_save_path,'best'))
		shutil.copyfile(f'{clmbr_info_path}', f'{clmbr_save_path}/info.json')
		with open(f'{clmbr_save_path}/config.json', 'w') as f:
			json.dump(config,f)
writer.close()
        
    