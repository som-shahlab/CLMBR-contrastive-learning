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

from sklearn.model_selection import ParameterGrid
#from torch.utils.data import DataLoader, Dataset


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
    default='/local-scratch/nigam/projects/jlemmon/cl-clmbr/experiments/main/artifacts/models/clmbr/contrastive_learn/models/gru_sz_800_do_0.1_cd_0_dd_0_lr_0.001_l2_0.01/',
    help='Base path for the best finetuned model.'
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
    default=20,
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
    '--device',
    type=str,
    default='cuda:0',
    help='Device to run torch model on.'
)

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
		elif self.sim == 'sumcos':
			# sum loss over patient timeline

			# iterate over patients
			# iterate over patient timeline
			# for day 1..D sum up cos(A_d,B_d)/self.temp
			pass
		else:
			return torch.dot(torch.flatten(x),torch.flatten(y)).unsqueeze(0).unsqueeze(0)

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
		elif self.pooler == 'sumcos':
			return embeds
		elif self.pooler == 'rand_day':
			outputs = torch.tensor([]).to(self.device)
			for i, e in enumerate(embeds):
				outputs = torch.concat((outputs, e[day_indices[i]]), 0)
			outputs = torch.reshape(outputs, (embeds.shape[0], 1, embeds.shape[-1]))
			return outputs

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
		# print(batch['pid'])
		# For patient timeline in batch get CLMBR embedding
		z1_embeds = self.clmbr_model.timeline_model(batch["rnn"])
		# print(z1_embeds.shape)
		# Run batch through CLMBR again to get different masked embedding for positive pairs
		z2_embeds = self.clmbr_model.timeline_model(batch["rnn"])
		# print(z2_embeds.shape)
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
		# print(pooled_embeds.shape)
		
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
    
def load_data(args, clmbr_hp):
	"""
	Load datasets from split csv files.
	"""

	data_path = f'{args.labelled_fpath}/hospital_mortality/pretrained/{args.encoder}_sz_{clmbr_hp["size"]}_do_{clmbr_hp["dropout"]}_cd_{clmbr_hp["code_dropout"]}_dd_{clmbr_hp["day_dropout"]}_lr_{clmbr_hp["lr"]}_l2_{clmbr_hp["l2"]}'

	
	train_pids = pd.read_csv(f'{data_path}/ehr_ml_patient_ids_train.csv')
	val_pids = pd.read_csv(f'{data_path}/ehr_ml_patient_ids_val.csv')

	train_days = pd.read_csv(f'{data_path}/day_indices_train.csv')
	val_days = pd.read_csv(f'{data_path}/day_indices_val.csv')

	train_labels = pd.read_csv(f'{data_path}/labels_train.csv')
	val_labels = pd.read_csv(f'{data_path}/labels_val.csv')

	train_data = (train_labels.to_numpy().flatten(),train_pids.to_numpy().flatten(),train_days.to_numpy().flatten())
	val_data = (val_labels.to_numpy().flatten(),val_pids.to_numpy().flatten(),val_days.to_numpy().flatten())

	return train_data, val_data
        
def finetune(args, model, train_dataset, val_dataset, lr, clmbr_save_path, clmbr_model_path):
	"""
	Finetune CLMBR model using linear layer.
	"""
	model.train()
	config = model.clmbr_model.config
	optimizer = optim.Adam([p for n, p in model.named_parameters() if p.requires_grad], lr=lr)
	best_val_loss = 9999999
	for e in range(args.epochs):
		model.train()
		train_loss = []
		with DataLoader(train_dataset, model.config['num_first'], is_val=False, batch_size=model.config["batch_size"], device=args.device) as train_loader:
			for batch in train_loader:
				# Skip batches that only consist of one patient
				if len(batch['pid']) == 1:
					continue
				else:
					optimizer.zero_grad()
					outputs = model(batch)
					loss = outputs["loss"]

					loss.backward()
					optimizer.step()
					train_loss.append(loss.item())
		print('Training loss:',  np.sum(train_loss))
		val_preds, val_lbls, val_losses = evaluate_model(args, model, val_dataset)
		scaled_val_loss = np.sum(val_losses)*model.temp
		
		os.makedirs(f'{clmbr_save_path}/{e}',exist_ok=True)
		torch.save(clmbr_model.state_dict(), os.path.join(clmbr_save_path,f'{e}/best'))
		shutil.copyfile(f'{clmbr_model_path}/info.json', f'{clmbr_save_path}/{e}/info.json')
		with open(f'{clmbr_save_path}/{e}/config.json', 'w') as f:
			json.dump(config,f)			
		if scaled_val_loss < best_val_loss:
			best_val_loss = scaled_val_loss
			best_model = copy.deepcopy(model.clmbr_model)
	return best_model, best_val_loss

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

if __name__ == '__main__':
    
	args = parser.parse_args()

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
	for i, clmbr_hp in enumerate(grid):
		
		clmbr_model_path = f'{args.pt_model_path}/{args.encoder}_sz_{clmbr_hp["size"]}_do_{clmbr_hp["dropout"]}_cd_{clmbr_hp["code_dropout"]}_dd_{clmbr_hp["day_dropout"]}_lr_{clmbr_hp["lr"]}_l2_{clmbr_hp["l2"]}'
		print(clmbr_model_path)
		best_ft_path = f"{args.model_path}/{args.encoder}_sz_{clmbr_hp['size']}_do_{clmbr_hp['dropout']}_cd_{clmbr_hp['code_dropout']}_dd_{clmbr_hp['day_dropout']}_lr_{clmbr_hp['lr']}_l2_{clmbr_hp['l2']}/best"
		os.makedirs(f"{best_ft_path}",exist_ok=True)
		best_val_loss = 9999999
		best_params = None
		for j, cl_hp in enumerate(cl_grid):
			print('finetuning model with params: ', cl_hp)
			clmbr_save_path = f"{args.model_path}/{args.encoder}_sz_{clmbr_hp['size']}_do_{clmbr_hp['dropout']}_cd_{clmbr_hp['code_dropout']}_dd_{clmbr_hp['day_dropout']}_lr_{clmbr_hp['lr']}_l2_{clmbr_hp['l2']}/bs_{cl_hp['batch_size']}_lr_{cl_hp['lr']}_temp_{cl_hp['temp']}_pool_{cl_hp['pool']}"
			print(clmbr_save_path)
			os.makedirs(f"{clmbr_save_path}",exist_ok=True)
			train_data, val_data = load_data(args, clmbr_hp)

			train_dataset = PatientTimelineDataset(args.extract_path + '/extract.db', 
											 args.extract_path + '/ontology.db', 
											 f'{clmbr_model_path}/info.json', 
											 train_data, 
											 train_data )
			
			val_dataset = PatientTimelineDataset(args.extract_path + '/extract.db', 
											 args.extract_path + '/ontology.db', 
											 f'{clmbr_model_path}/info.json', 
											 val_data, 
											 val_data )

			clmbr_model = ehr_ml.clmbr.CLMBR.from_pretrained(clmbr_model_path, args.device)
			# Modify CLMBR config settings
			clmbr_model.config["model_dir"] = clmbr_save_path
			clmbr_model.config["batch_size"] = cl_hp['batch_size']
			clmbr_model.config["epochs_per_cycle"] = args.epochs
			clmbr_model.config["warmup_epochs"] = 1

			config = clmbr_model.config

			clmbr_model.unfreeze()
			# Get contrastive learning model 
			model = ContrastiveLearn(clmbr_model, 2, cl_hp['pool'], cl_hp['temp'], args.device).to(args.device)
			model.train()

			# Run finetune procedure
			# trainer = Trainer(model)
			# trainer.train(dataset)
			clmbr_model, val_loss = finetune(args, model, train_dataset, val_dataset, float(cl_hp['lr']), clmbr_save_path, clmbr_model_path)
			clmbr_model.freeze()
			if val_loss < best_val_loss:
				print('Saving as best finetuned model...')
				best_val_loss = val_loss
				best_params = cl_hp
				
				torch.save(clmbr_model.state_dict(), os.path.join(best_ft_path,'best'))
				shutil.copyfile(f'{clmbr_model_path}/info.json', f'{best_ft_path}/info.json')
				with open(f'{best_ft_path}/config.json', 'w') as f:
					json.dump(config,f)
				with open(f"{best_ft_path}/hyperparams.yml", 'w') as file: # fix format of dump
					yaml.dump(best_params,file)
				
			# Save model and save info and config to new model directory for downstream evaluation
			torch.save(clmbr_model.state_dict(), os.path.join(clmbr_save_path,'best'))
			shutil.copyfile(f'{clmbr_model_path}/info.json', f'{clmbr_save_path}/info.json')
			with open(f'{clmbr_save_path}/config.json', 'w') as f:
				json.dump(config,f)
        
    