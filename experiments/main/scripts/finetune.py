import os
import json
import argparse
import random
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
#from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument(
    '--task',
    type=str,
    required=True,
    help='Name of task to label. Accepted tasks are:\nhospital_mortality\nLOS_7\nicu_admission\nreadmission_30'
)
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
    
    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)
        
    def forward(self, x, y):
        return self.cos(x, y) / self.temp

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
		self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	def forward(self, batch, is_train=True):
		# For patient timeline in batch get CLMBR embedding
		z1_embeds = self.clmbr_model.timeline_model(batch["rnn"])

		# Run batch through CLMBR again to get different masked embedding for positive pairs
		z2_embeds = self.clmbr_model.timeline_model(batch["rnn"])

		# Flatten embeddings
		z1_flat_embeds = z1_embeds.view((-1, z1_embeds.shape[-1]))
		z2_flat_embeds = z2_embeds.view((-1, z2_embeds.shape[-1]))

		# Use pooler to get target embeddings
		z1_target_embeds = self.pooler(z1_flat_embeds)
		z2_target_embeds = self.pooler(z2_flat_embeds)

		
		# Reshape pooled embeds to BATCH_SIZE X 2 X EMBEDDING_SIZE
		# First column is z1, second column is z2
		pooled_embeds = torch.stack((z1_target_embeds, z2_target_embeds))
		pooled_embeds = pooled_embeds.view(1 , 2, pooled_embeds.size(-1))

		# If [CLS] pooling used, run pooled embeddings through linear layer
		if self.pooler.pooler == 'cls':
			if args.mlp_train == 1 and is_train:
				pooled_embeds = self.linear(pooled_embeds)
			
		z1, z2 = pooled_embeds[:,0], pooled_embeds[:,1]
		
		# Get cosine similarity
		cos_sim = self.sim(z1.unsqueeze(1), z2.unsqueeze(0))
		# Generate labels
		labels = torch.arange(cos_sim.size(0)).long().to(self.device)
		
		return cos_sim, labels
    
def load_data(args):
    """
    Load datasets from split csv files.
    """
    data_path = f'{args.labelled_fpath}/baseline/{args.encoder}_{args.size}_{args.dropout}_{args.lr}/{args.task}'
    
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
	#val_loader = DataLoader(dataset, batch_size=args.batch_size, is_val=True, seed=args.seed, device=args.device)

	optimizer = optim.Adam([p for n, p in model.named_parameters() if p.requires_grad], lr=args.lr)
	criterion = nn.CrossEntropyLoss()

	pbar = tqdm(total=args.epochs * dataset.num_batches(args.batch_size, False))
	
	step_train_loss = []

	for e in range(args.epochs):
		for step, batch in enumerate(tqdm(train_loader, desc='Iteration')):
			optimizer.zero_grad()
			logits, labels = model(batch)
			print('logits', logits)
			print('label', labels)
			loss = criterion(logits, labels)
			if args.gradient_accumulation > 1:
				loss = loss / args.gradient_accumulation
			print('loss', loss)
			loss.backward()
			if (step + 1) % args.gradient_accumulation == 0:
				optimizer.step()
				model.zero_grad()
			step_train_loss.append(loss.item())
			#todo validation check
			pbar.update(1)
	return model.clmbr_model

def validate(args, model, dataset):
	config = model.clmbr_model.config
	val_loader = DataLoader(dataset, config['num_first'], is_val=True, batch_size=args.batch_size, seed=args.seed, device=args.device)
	criterion = nn.CrossEntropyLoss()
	
	
	
	

if __name__ == '__main__':
    
	args = parser.parse_args()

	torch.manual_seed(args.seed)

	clmbr_model_path = f"{args.pt_model_path}/{args.encoder}_{args.size}_{args.dropout}_{args.lr}"
	print(clmbr_model_path)
	clmbr_save_path = f"{args.model_path}/contrastive_learn/{args.encoder}_{args.size}_{args.dropout}_{args.cl_lr}"
	train_data, val_data, test_data = load_data(args)

	dataset = PatientTimelineDataset(args.extract_path + '/extract.db', 
									 args.extract_path + '/ontology.db', 
									 f'{clmbr_model_path}/info.json', 
									 train_data, 
									 val_data )

	clmbr_model = ehr_ml.clmbr.CLMBR.from_pretrained(clmbr_model_path, args.device)
	clmbr_model.config["model_dir"] = clmbr_save_path

	model = ContrastiveLearn(clmbr_model, 2, args.pooler, args.temp, args.device).to(args.device)

	log_path = clmbr_save_path + '/logs'
	model = finetune(args, model, dataset)
	model.freeze()
	val = validate(args, model, dataset)
	model.clmbr_model.save(clmbr_save_path)
        
    