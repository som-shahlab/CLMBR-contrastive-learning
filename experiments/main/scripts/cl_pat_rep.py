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
    default='/local-scratch/nigam/projects/jlemmon/cl-clmbr/experiments/main/artifacts/models/clmbr/cl_ete/models',
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
    default="/local-scratch/nigam/projects/jlemmon/cl-clmbr/experiments/main/data/labelled_data/hospital_mortality/pretrained/gru_sz_800_do_0.1_cd_0_dd_0_lr_0.001_l2_0.01",
    help='Base path for labelled data directory'
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
    default=512,
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
	'--gradient_accumulation',
	type=int,
	default=1,
	help='Value to divide loss by for gradient accumulation.'
)

parser.add_argument(
	'--mlp_train',
	type=int,
	default=1,
	help='Use MLP layer for [CLS] pooling during training only.'
)

parser.add_argument(
	'--patience',
	type=int,
	default=10,
	help='Number of epochs to wait before triggering early stopping.'
)

parser.add_argument(
    '--device',
    type=str,
    default='cuda:0',
    help='Device to run torch model on.'
)

class MLPLayer(nn.Module):
	"""
	Linear classifier layer.
	"""
	def __init__(self, size):
		super().__init__()
		self.dense = nn.Linear(size, size)
		self.activation = nn.Tanh()
		
	def forward(self, features):
		x = self.dense(features)
		x = self.activation(x)
		
		return x

class Similarity(nn.Module):
	"""
	Cosine similarity with temperature value for embedding similarity calculation.
	"""

	def __init__(self, temp, sim='cos'):
		super().__init__()
		self.temp = temp
		self.sim = sim
		self.cos = nn.CosineSimilarity(dim=-1)

	def forward(self, x, y):
		if self.sim == 'cos':
			return self.cos(x, y) / self.temp
		else:
			return torch.dot(torch.flatten(x),torch.flatten(y)).unsqueeze(0).unsqueeze(0)

class Pooler(nn.Module):
	"""
	Pooler module to get the pooled embedding value. 
	"""
	def __init__(self, pooler, device=None):
		super().__init__()
		self.pooler = pooler
		self.device = device if device is not None else 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
	def forward(self, embeds, day_indices=None):
		# Only CLS style pooling for now
		if self.pooler == 'mean_rep':
			# Generate positive pairs using mean embedding of both augmented timelines
			return torch.mean(embeds,1,True)
		elif self.pooler == 'rand_day':
			# Generate positive pairs using the same random day index for both augmented timelines
			outputs = torch.tensor([]).to(self.device)
			for i, e in enumerate(embeds):
				outputs = torch.concat((outputs, e[day_indices[i]]), 0)
			outputs = torch.reshape(outputs, (embeds.shape[0], 1, embeds.shape[-1]))
			return outputs
		elif self.pooler == 'diff_pat':
			# Generate positive pairs using two random day embeddings from one patient timeline
			z1_out = torch.tensor([]).to(self.device)
			z2_out = torch.tensor([]).to(self.device)
			skipped = 0
			for i, e in enumerate(embeds):
				if e.shape[0] < 4:
					skipped += 1
				else:
					z1_ind = np.random.randint(1, e.shape[0]-2)
					z2_ind = np.random.randint(z1_ind+1, e.shape[0]-1)
					z1_out = torch.concat((z1_out, e[z1_ind]), 0)
					z2_out = torch.concat((z2_out, e[z2_ind]), 0)
			z1_out = torch.reshape(z1_out, (embeds.shape[0]-skipped, 1, embeds.shape[-1]))
			z2_out = torch.reshape(z2_out, (embeds.shape[0]-skipped, 1, embeds.shape[-1]))
			return z1_out, z2_out
		elif self.pooler == 'trivial':
			# Generate positive pairs using two consecutive day embeddings
			# Used for sanity check on task difficulty
			z1_out = torch.tensor([]).to(self.device)
			z2_out = torch.tensor([]).to(self.device)
			for i, e in enumerate(embeds):
				idx = np.random.randint(0,e.shape[0]-1)
				z1_out = torch.concat((z1_out, e[idx]), 0)
				z2_out = torch.concat((z2_out, e[idx+1]), 0)
			z1_out = torch.reshape(z1_out, (embeds.shape[0], 1, embeds.shape[-1]))
			z2_out = torch.reshape(z2_out, (embeds.shape[0], 1, embeds.shape[-1]))
			return z1_out, z2_out

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
		self.temp = temp
		self.sim = Similarity(temp)
		self.criterion = nn.CrossEntropyLoss()
		self.device = device if device is not None else 'cuda:0' if torch.cuda.is_available() else 'cpu'

	def forward(self, batch, is_train=True):
		outputs = dict()

		# For patient timeline in batch get CLMBR embedding
		z1_embeds = self.clmbr_model.timeline_model(batch["rnn"])

		# Run batch through CLMBR again to get different masked embedding for positive pairs
		z2_embeds = self.clmbr_model.timeline_model(batch["rnn"])
		
		# Flatten embeddings
		rand_day_indices = None
		if self.pooler.pooler == 'rand_day':
			rand_day_indices = []
			for di in batch['day_index']:
				rand_day_indices.append(random.choice(di))
		# Use pooler to get target embeddings
		z1_target_embeds = self.pooler(z1_embeds, rand_day_indices)
		z2_target_embeds = self.pooler(z2_embeds, rand_day_indices)

		# Reshape pooled embeds to BATCH_SIZE X 2 X EMBEDDING_SIZE
		# First column is z1, second column is z2
		pooled_embeds = torch.concat((z1_target_embeds, z2_target_embeds), axis=0)
		pooled_embeds = pooled_embeds.view(len(batch['pid']), 2, pooled_embeds.size(-1))
		
		pooled_embeds = self.linear(pooled_embeds)
		z1, z2 = pooled_embeds[:,0], pooled_embeds[:,1]

		# Get cosine similarity
		cos_sim = self.sim(z1.unsqueeze(1), z2.unsqueeze(0))

		# Generate labels
		labels = torch.arange(cos_sim.size(0)).long().to(self.device)
		
		# Compute loss
		outputs['loss'] = self.criterion(cos_sim,labels)
		outputs['preds'] = cos_sim
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
	
def load_data(args, clmbr_hp):
	"""
	Load datasets from split csv files.
	"""

	data_path = f'{args.labelled_fpath}'

	
	train_pids = pd.read_csv(f'{data_path}/ehr_ml_patient_ids_train.csv')
	val_pids = pd.read_csv(f'{data_path}/ehr_ml_patient_ids_val.csv')

	train_days = pd.read_csv(f'{data_path}/day_indices_train.csv')
	val_days = pd.read_csv(f'{data_path}/day_indices_val.csv')

	train_labels = pd.read_csv(f'{data_path}/labels_train.csv')
	val_labels = pd.read_csv(f'{data_path}/labels_val.csv')

	train_data = (train_labels.to_numpy().flatten(),train_pids.to_numpy().flatten(),train_days.to_numpy().flatten())
	val_data = (val_labels.to_numpy().flatten(),val_pids.to_numpy().flatten(),val_days.to_numpy().flatten())

	return train_data, val_data
        
def train(args, model, train_dataset, val_dataset, lr, clmbr_save_path, clmbr_info_path, bl_int, cl_int):
	"""
	Train CLMBR model using CL objective.
	"""
	model.train()
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
		with DataLoader(train_dataset, model.config['num_first'], is_val=False, batch_size=model.config["batch_size"], device=args.device) as train_loader:
			for batch in train_loader:
				# Skip batches that only consist of one patient
				if len(batch['pid']) == 1:
					continue
				else:
					optimizer.zero_grad()
					outputs = model(batch)
					train_preds.extend(list(outputs['preds'].detach().clone().cpu().numpy()))
					train_lbls.extend(list(outputs['labels'].detach().clone().cpu().numpy()))
					loss = outputs["loss"]

					loss.backward()
					optimizer.step()
					train_loss.append(loss.item())
		print('Training loss:',  np.sum(train_loss))
		model_train_loss.append(np.sum(train_loss))
		
		# evaluate on validation set
		val_preds, val_lbls, val_losses = evaluate_model(args, model, val_dataset)
		scaled_val_loss = np.sum(val_losses)*model.temp
		model_val_loss.append(np.sum(val_losses))
		
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
		if scaled_val_loss < best_val_loss:
			best_val_loss = scaled_val_loss
			best_epoch = e
			best_model = copy.deepcopy(model.clmbr_model)
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

def evaluate_model(args, model, dataset):
	model.eval()
	
	criterion = nn.CrossEntropyLoss()
	
	preds = []
	lbls = []
	losses = []
	with torch.no_grad():
		with DataLoader(dataset, model.config['num_first'], is_val=True, batch_size=model.config['batch_size'], seed=args.seed, device=args.device) as eval_loader:
			for batch in eval_loader:
				outputs = model(batch)
				loss = outputs["loss"]
				losses.append(loss.item())
				preds.extend(list(outputs['preds'].cpu().numpy()))
				lbls.extend(list(outputs['labels'].cpu().numpy()))
	print('Validation loss:',  np.sum(losses))
	return preds, lbls, losses

def get_config(hp):
    config = {'b1': 0.9,
            'b2': 0.999,
            'batch_size': hp['batch_size'],
            'code_dropout': hp['code_dropout'],
            'day_dropout': hp['day_dropout'],
            'dropout': hp['dropout'],
            'e': 1e-08,
            'encoder_type': hp['encoder_type'],
            'epochs_per_cycle': 1,
            'eval_batch_size': hp['batch_size'],
            'l2': hp['l2'],
            'lr': hp['lr'],
            'model_dir': '',
            'num_first': 9262,
            'num_second': 10044,
            'rnn_layers': 1,
            'size': hp['size'],
            'tied_weights': True,
            'warmup_epochs': hp['warmup_epochs']}
    return config

if __name__ == '__main__':
    
	args = parser.parse_args()

	torch.manual_seed(args.seed)
	
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
					f"{os.path.join(args.hparams_fpath,'cl-ete')}.yml",
					'r'
				),
				Loader=yaml.FullLoader
			)
		)
	)
	for i, clmbr_hp in enumerate(grid):
		print('Initialized CLMBR model with params: ', clmbr_hp)
		clmbr_info_path = f'{args.pt_info_path}/info.json'
		with open(clmbr_info_path) as f:
			info = json.load(f)
		config = get_config(clmbr_hp)
		best_val_loss = 9999999
		best_params = None
		for j, cl_hp in enumerate(cl_grid):
			print('Training model with CL params: ', cl_hp)
			config["batch_size"] = cl_hp['batch_size']
			bl_model_str = f"{args.encoder}_sz_{clmbr_hp['size']}_do_{clmbr_hp['dropout']}_cd_{clmbr_hp['code_dropout']}_dd_{clmbr_hp['day_dropout']}_lr_{clmbr_hp['lr']}_l2_{clmbr_hp['l2']}"
			cl_model_str = f"bs_{cl_hp['batch_size']}_lr_{cl_hp['lr']}_temp_{cl_hp['temp']}_pool_{cl_hp['pool']}"
			clmbr_save_path = f"{args.model_path}/{bl_model_str}_{cl_model_str}"
			print(clmbr_save_path)
			os.makedirs(f"{clmbr_save_path}",exist_ok=True)
			train_data, val_data = load_data(args, clmbr_hp)

			train_dataset = PatientTimelineDataset(args.extract_path + '/extract.db', 
											 args.extract_path + '/ontology.db', 
											 clmbr_info_path, 
											 train_data, 
											 train_data )
			
			val_dataset = PatientTimelineDataset(args.extract_path + '/extract.db', 
											 args.extract_path + '/ontology.db', 
											 clmbr_info_path, 
											 val_data, 
											 val_data )
			config["model_dir"] = clmbr_save_path
			clmbr_model = ehr_ml.clmbr.CLMBR(config, info).to(torch.device(args.device))
			# Modify CLMBR config settings
			
			config = clmbr_model.config

			clmbr_model.unfreeze()
			# Get contrastive learning model 
			model = ContrastiveLearn(clmbr_model, 2, cl_hp['pool'], cl_hp['temp'], args.device).to(args.device)
			model.train()

			# Run finetune procedure
			clmbr_model, val_loss, val_df = train(args, model, train_dataset, val_dataset, float(cl_hp['lr']), clmbr_save_path, clmbr_info_path, i, j)
			writer.flush()
			clmbr_model.freeze()
			if val_loss < best_val_loss:
				print('Saving as best trained model...')
				best_val_loss = val_loss
				best_params = cl_hp
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
        
    