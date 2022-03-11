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
    '--probe_path',
    type=str,
    default='/local-scratch/nigam/projects/jlemmon/cl-clmbr/experiments/main/artifacts/models/probes/baseline/models',
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
    '--device',
    type=str,
    default='cuda:0',
    help='Device to run torch model on.'
)


class LinearProbe(nn.module):
	def __init__(self, clmbr_model, size):
		super().__init__()
		self.clmbr_model = clmbr_model
		self.config = clmbr_moel.config
		self.dense = nn.Linear(size,1)
		self.activation = nn.Sigmoid()
		
		self.criterion = nn.BCELoss()
	
	def forward(self, x, y):
		features = self.clmbr_model(x)
		preds = self.dense(features)
		preds = self.activation(preds)
		
		outputs['preds'] = preds
		outputs['loss'] = self.criterion(preds, y)
		
		return outputs
	
	def freeze_clmbr(self):
		self.clmbr_model.freeze()
	
	def unfreeze_clmbr(self):
		self.clmbr_model.unfreeze()

	
def load_datasets(args, clmbr_hp):
    """
    Load datasets from split csv files.
    """
    data_path = f'{args.labelled_fpath}/{args.task}/pretrained/{args.encoder}_sz_{clmbr_hp["size"]}_do_{clmbr_hp["dropout"]}_lr_{clmbr_hp["lr"]}_l2_{clmbr_hp["l2"]}'
    
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
											 train_data )
	val_dataset = PatientTimelineDataset(args.extract_path + '/extract.db', 
										 args.extract_path + '/ontology.db', 
										 f'{clmbr_model_path}/info.json', 
										 val_data, 
										 val_data )
	test_dataset = PatientTimelineDataset(args.extract_path + '/extract.db', 
										 args.extract_path + '/ontology.db', 
										 f'{clmbr_model_path}/info.json', 
										 test_data, 
										 test_data )
    
    return train_dataset, val_dataset, test_dataset


def train_probe(args, model, dataset):
	train_loader = DataLoader(dataset, model.config['num_first'], is_val=False, batch_size=model.config["batch_size"], seed=args.seed, device=args.device)
	model.train()
	model.freeze_clmbr()
	optimizer = optim.Adam([p for n, p in model.named_parameters() if p.requires_grad], lr=args.lr)
	
	for e in range(args.epochs):
		for batch in train_loader:
			
			optimizer.zero_grad()
			outputs = model(batch)
			loss = outputs['loss']
			
			loss.backward()
			print('training loss', loss.item())
	
	return model

def evaluate_probe(args, model, dataset):
	eval_loader = DataLoader(dataset, model.config['num_first'], is_val=True, batch_size=model.config['batch_size'], seed=args.seed, device=args.device)
	model.eval()
	
	preds = []
	losses = []
	with torch.no_grad():
		for batch in eval_loader:
			outputs = model(batch)
			losses.append(outputs['loss'])
			preds.append(outputs['preds'])
	
	return preds, losses
			
	
if __name__ == '__main__':
	args = parser.parse_args()
	
	torch.manual_seed(args.seed)
	
	tasks = ['hospital_mortality', 'LOS_7', 'icu_admission', 'readmission_30']
	
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
	
	for i, clmbr_hp in enumerate(grid):
		
		for task in tasks:
		
			clmbr_model_path = f'{args.pt_model_path}/{args.encoder}_sz_{clmbr_hp["size"]}_do_{clmbr_hp["dropout"]}_lr_{clmbr_hp["lr"]}_l2_{clmbr_hp["l2"]}'
			print(clmbr_model_path)


			train_dataset, val_dataset, test_dataset = load_datasets(args, clmbr_hp)

			probe_save_path = f'{args.probe_path}/baseline/{args.encoder}_sz_{clmbr_hp["size"]}_do_{clmbr_hp["dropout"]}_lr_{clmbr_hp["lr"]}_l2_{clmbr_hp["l2"]}/{task}''
			os.makedirs(f"{clmbr_save_path}",exist_ok=True)
			
			clmbr_model = ehr_ml.clmbr.CLMBR.from_pretrained(clmbr_model_path, args.device)
			clmbr_model.freeze()
			
			probe_model = LinearProbe(clmbr_model, clmbr_hp['size'])
			
			probe_model = train_probe(args, probe_model, dataset)
			
			val_preds, val_losses = evaluate_probe(args, probe_model, val_dataset)
			print(val_preds)
			print(val_losses)
			
			test_preds, test_losses = evaluate_probe(args, probe_model, test_dataset)
			print(test_preds)
			print(test_losses)
		