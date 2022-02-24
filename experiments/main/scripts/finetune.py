import os
import json
import argparse
import shutil
import yaml
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
    default='2015-12-31',
    help='End date of training ids.'
)

parser.add_argument(
    '--val_end_date',
    type=str,
    default='2016-07-01',
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
    default=1,
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
    default='cuda:3',
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
		else:
			return torch.dot(torch.flatten(x),torch.flatten(y)).unsqueeze(0).unsqueeze(0)

class Pooler(nn.Module):
	"""
	Pooler module to get the pooled embedding value. [cls] is equivalent to the final hidden layer embedding.
	"""
	def __init__(self, pooler):
		super().__init__()
		self.pooler = pooler
        
	def forward(self, outputs):
		# Only CLS style pooling for now
		if self.pooler == 'cls':
			return outputs[-1]
		elif self.pooler == 'sumcos':
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
		self.pooler = Pooler(pooler)
		self.sim = Similarity(temp)
		self.criterion = nn.CrossEntropyLoss()
		self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	def forward(self, batch, is_train=True):
		outputs = dict()
		# For patient timeline in batch get CLMBR embedding
		z1_embeds = [self.clmbr_model.timeline_model(b["rnn"]) for b in batch]

		# Run batch through CLMBR again to get different masked embedding for positive pairs
		z2_embeds = [self.clmbr_model.timeline_model(b["rnn"]) for b in batch]

		# Flatten embeddings
		z1_flat_embeds = [z1_embed.view((-1, z1_embed.shape[-1])) for z1_embed in z1_embeds]
		z2_flat_embeds = [z2_embed.view((-1, z2_embed.shape[-1])) for z2_embed in z2_embeds]

		# Use pooler to get target embeddings
		z1_target_embeds = [self.pooler(z1_flat_embed) for z1_flat_embed in z1_flat_embeds]
		z1_target_embeds = torch.stack(z1_target_embeds)
		z2_target_embeds = [self.pooler(z2_flat_embed) for z2_flat_embed in z2_flat_embeds]
		z2_target_embeds = torch.stack(z2_target_embeds)

		# Reshape pooled embeds to BATCH_SIZE X 2 X EMBEDDING_SIZE
		# First column is z1, second column is z2
		pooled_embeds = torch.concat((z1_target_embeds, z2_target_embeds), axis=0)
		pooled_embeds = pooled_embeds.view(len(batch), 2, pooled_embeds.size(-1))

		# If [CLS] pooling used, run pooled embeddings through linear layer
		if self.pooler.pooler == 'cls':
			pooled_embeds = self.linear(pooled_embeds)
		z1, z2 = pooled_embeds[:,0], pooled_embeds[:,1]

		# Get cosine similarity
		cos_sim = self.sim(z1.unsqueeze(1), z2.unsqueeze(0))

		# Generate labels
		labels = torch.arange(cos_sim.size(0)).long().to(self.device)
		
		# Compute loss
		outputs['loss'] = self.criterion(cos_sim,labels)

		return outputs
    
def load_data(args, clmbr_hp):
    """
    Load datasets from split csv files.
    """
    data_path = f'{args.labelled_fpath}/hospital_mortality/pretrained/{args.encoder}_sz_{clmbr_hp["size"]}_do_{clmbr_hp["dropout"]}_lr_{clmbr_hp["lr"]}_l2_{clmbr_hp["l2"]}'
    
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
    
    return train_data, val_data, test_data
        
def finetune(args, model, dataset):
	"""
	Finetune CLMBR model using linear layer.
	"""
	model.train()
	config = model.clmbr_model.config
	train_loader = DataLoader(dataset, config['num_first'], is_val=False, batch_size=args.batch_size, seed=args.seed, device=args.device)
	
	optimizer = optim.Adam([p for n, p in model.named_parameters() if p.requires_grad], lr=args.lr)
	criterion = nn.CrossEntropyLoss()
	
	step_train_loss = []

	for e in range(args.epochs):
		batch = []
		for i in range(args.batch_size):
			batch.append(next(train_loader))
			
		optimizer.zero_grad()
		outputs = model(batch)
		loss = outputs["loss"]

		print('train loss', loss)
		loss.backward()
		step_train_loss.append(loss.item())
	
	# train_loader threads hanging, iterate through batches to terminate
	# temporary until find reason for number of batches to be so large/find way to terminate threads early
	print('Terminating data loader threads...')
	for i, batch in enumerate(train_loader):
		pass

	return model.clmbr_model

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
					f"{os.path.join(args.hparams_fpath,'cl')}.yml",
					'r'
				),
				Loader=yaml.FullLoader
			)
		)
	)
	for i, clmbr_hp in enumerate(grid):
		
		clmbr_model_path = f'{args.pt_model_path}/{args.encoder}_sz_{clmbr_hp["size"]}_do_{clmbr_hp["dropout"]}_lr_{clmbr_hp["lr"]}_l2_{clmbr_hp["l2"]}'
		print(clmbr_model_path)
		
		for j, cl_hp in enumerate(cl_grid):
			clmbr_save_path = f"{args.model_path}/{args.encoder}_sz_{clmbr_hp['size']}_do_{clmbr_hp['dropout']}_lr_{clmbr_hp['lr']}_l2_{clmbr_hp['l2']}/bs_{cl_hp['batch_size']}_lr_{cl_hp['lr']}_temp_{cl_hp['temp']}_pool_{cl_hp['pool']}"
			print(clmbr_save_path)
			os.makedirs(f"{clmbr_save_path}",exist_ok=True)
			train_data, val_data, test_data = load_data(args, clmbr_hp)

			dataset = PatientTimelineDataset(args.extract_path + '/extract.db', 
											 args.extract_path + '/ontology.db', 
											 f'{clmbr_model_path}/info.json', 
											 train_data, 
											 val_data ).to(args.device)

			clmbr_model = ehr_ml.clmbr.CLMBR.from_pretrained(clmbr_model_path, args.device)
			# Modify CLMBR config settings
			clmbr_model.config["model_dir"] = clmbr_save_path
			clmbr_model.config["batch_size"] = cl_hp['batch_size']
			clmbr_model.config["e"] = cl_hp['lr']
			clmbr_model.config["epochs_per_cycle"] = args.epochs
			clmbr_model.config["warmup_epochs"] = 1

			config = clmbr_model.config

			clmbr_model.unfreeze()
			# Get contrastive learning model 
			model = ContrastiveLearn(clmbr_model, 2, cl_hp['pool'], cl_hp['temp'], args.device).to(args.device)
			model.train()

			# Run finetune procedure
			clmbr_model = finetune(args, model, dataset)
			clmbr_model.freeze()

			# Save model and save info and config to new model directory for downstream evaluation
			torch.save(clmbr_model.state_dict(), os.path.join(clmbr_save_path,'best'))
			shutil.copyfile(f'{clmbr_model_path}/info.json', f'{clmbr_save_path}/info.json')
			with open(f'{clmbr_save_path}/config.json', 'w') as f:
				json.dump(config,f)
        
    