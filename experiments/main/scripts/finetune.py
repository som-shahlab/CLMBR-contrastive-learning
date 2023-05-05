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
from tqdm import tqdm

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
    default=50,
    help='Number of training epochs.'
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
    default='trivial',
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
	default=5,
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
        
	def forward(self, embeds, embeds_2=None, day_indices=None, pids=None):
		if self.pooler == 'mean_rep':
			# Generate positive pairs using mean embedding of both augmented timelines
			z1_mean_embeds = torch.mean(embeds,1,True)
			z2_mean_embeds = torch.mean(embeds_2,1,True)
			z1_out = torch.tensor([]).to(self.device)
			z2_out = torch.tensor([]).to(self.device)
			labels = []
			pat_info_df = pd.DataFrame()
			for i, e in enumerate(z1_mean_embeds):
				l_pair_id = []
				r_pair_id = []
				l_pair_idx = []
				r_pair_idx = []
				l_pair_max_idx = []
				r_pair_max_idx = []
				
				l_pair_id.append(pids[i])
				r_pair_id.append(pids[i])
				l_pair_idx.append('n/a')
				r_pair_idx.append('n/a')
				l_pair_max_idx.append(len(embeds[i]))
				r_pair_max_idx.append(len(embeds[i]))
				z1_out = torch.concat((z1_out, torch.reshape(e, (1,1,embeds.shape[-1]))), 0)
				z2_out = torch.concat((z2_out, torch.reshape(z2_mean_embeds[i], (1,1,embeds.shape[-1]))), 0)
				labels.append(1)
				neg_embed_idx = [j for j in list(np.arange(len(z2_mean_embeds))) if j != i]
				if len(neg_embed_idx) > 5:
					neg_embed_idx = np.random.choice(neg_embed_idx, 5)
				for j in neg_embed_idx:				
					neg_embeds = z2_mean_embeds[j]
					z1_out = torch.concat((z1_out, torch.reshape(e, (1,1,embeds.shape[-1]))), 0)
					neg_idx = np.random.randint(0, neg_embeds.shape[0])
					z2_out = torch.concat((z2_out, torch.reshape(neg_embeds[neg_idx], (1,1,embeds.shape[-1]))), 0)
					labels.append(0)
					l_pair_id.append(pids[i])
					r_pair_id.append(pids[j])
					l_pair_idx.append('n/a')
					r_pair_idx.append('n/a')
					l_pair_max_idx.append(len(embeds[i]))
					r_pair_max_idx.append(len(embeds_2[j]))
				df = pd.DataFrame({'left_id':l_pair_id, 'left_idx':l_pair_idx, 'left_max_idx':l_pair_max_idx, 'right_id':r_pair_id, 'right_idx':r_pair_idx, 'right_max_idx':r_pair_max_idx})
				pat_info_df =pd.concat((pat_info_df,df))
			labels = torch.tensor(labels).float().to(self.device)
			return z1_out, z2_out, labels, pat_info_df
				
		elif self.pooler == 'rand_day':
			# Generate positive pairs using the same random day index for both augmented timelines
			outputs = torch.tensor([]).to(self.device)
			z1_out = torch.tensor([]).to(self.device)
			z2_out = torch.tensor([]).to(self.device)
			pat_info_df = pd.DataFrame()
			labels = []
			for i, e in enumerate(embeds):
				l_pair_id = []
				r_pair_id = []
				l_pair_idx = []
				r_pair_idx = []
				l_pair_max_idx = []
				r_pair_max_idx = []
				
				l_pair_id.append(pids[i])
				r_pair_id.append(pids[i])
				l_pair_idx.append(day_indices[i])
				r_pair_idx.append(day_indices[i])
				l_pair_max_idx.append(len(e))
				r_pair_max_idx.append(len(e))
				z1_out = torch.concat((z1_out, torch.reshape(e[day_indices[i]], (1,1,embeds.shape[-1]))), 0)
				z2_out = torch.concat((z2_out, torch.reshape(embeds_2[i][day_indices[i]],(1,1,embeds.shape[-1]))),0)
				labels.append(1)
				neg_embed_idx = [j for j in list(np.arange(len(embeds))) if j != i]
				if len(neg_embed_idx) > 5:
					neg_embed_idx = np.random.choice(neg_embed_idx, 5)
				for j in neg_embed_idx:		
					neg_embeds = embeds_2[j]
					z1_out = torch.concat((z1_out, torch.reshape(e[day_indices[i]], (1,1,embeds.shape[-1]))), 0)
					neg_idx = np.random.randint(0, neg_embeds.shape[0])
					z2_out = torch.concat((z2_out, torch.reshape(neg_embeds[neg_idx], (1,1,embeds.shape[-1]))), 0)
					labels.append(0)
					l_pair_id.append(pids[i])
					r_pair_id.append(pids[j])
					l_pair_idx.append(day_indices[i])
					r_pair_idx.append(neg_idx)
					l_pair_max_idx.append(len(e))
					r_pair_max_idx.append(len(neg_embeds))
				df = pd.DataFrame({'left_id':l_pair_id, 'left_idx':l_pair_idx, 'left_max_idx':l_pair_max_idx, 'right_id':r_pair_id, 'right_idx':r_pair_idx, 'right_max_idx':r_pair_max_idx})
				pat_info_df =pd.concat((pat_info_df,df))
			labels = torch.tensor(labels).float().to(self.device)
			return z1_out, z2_out, labels, pat_info_df
		elif self.pooler == 'diff_pat':
			# Generate positive pairs using two random day embeddings from one patient timeline
			z1_out = torch.tensor([]).to(self.device)
			z2_out = torch.tensor([]).to(self.device)
			pat_info_df = pd.DataFrame()
			skipped = 0
			labels = []
			for i, e in enumerate(embeds):
				if e.shape[0] < 4:
					skipped += 1
				else:
					l_pair_id = []
					r_pair_id = []
					l_pair_idx = []
					r_pair_idx = []
					l_pair_max_idx = []
					r_pair_max_idx = []
					
					z1_ind = np.random.randint(1, e.shape[0]-2)
					z2_ind = np.random.randint(z1_ind+1, e.shape[0]-1)
					z1_out = torch.concat((z1_out, torch.reshape(e[z1_ind], (1,1,embeds.shape[-1]))), 0)
					z2_out = torch.concat((z2_out, torch.reshape(e[z2_ind], (1,1,embeds.shape[-1]))), 0)
					labels.append(1)
					l_pair_id.append(pids[i])
					r_pair_id.append(pids[i])
					l_pair_idx.append(z1_ind)
					r_pair_idx.append(z2_ind)
					l_pair_max_idx.append(len(e))
					r_pair_max_idx.append(len(e))
					neg_embed_idx = [j for j in list(np.arange(len(embeds))) if j != i]
					if len(neg_embed_idx) > 5:
						neg_embed_idx = np.random.choice(neg_embed_idx, 5)
					for j in neg_embed_idx:				
						neg_embeds = embeds[j]
						z1_out = torch.concat((z1_out, torch.reshape(e[z1_ind], (1,1,embeds.shape[-1]))), 0)
						neg_idx = np.random.randint(0, neg_embeds.shape[0])
						z2_out = torch.concat((z2_out, torch.reshape(neg_embeds[neg_idx], (1,1,embeds.shape[-1]))), 0)
						labels.append(0)
						l_pair_id.append(pids[i])
						r_pair_id.append(pids[j])
						l_pair_idx.append(z1_ind)
						r_pair_idx.append(neg_idx)
						l_pair_max_idx.append(len(e))
						r_pair_max_idx.append(len(neg_embeds))
					df = pd.DataFrame({'left_id':l_pair_id, 'left_idx':l_pair_idx, 'left_max_idx':l_pair_max_idx, 'right_id':r_pair_id, 'right_idx':r_pair_idx, 'right_max_idx':r_pair_max_idx})
					pat_info_df = pd.concat((pat_info_df,df))
			labels = torch.tensor(labels).float().to(self.device)
			if z1_out.shape[0] == 0:
				return None, None, None, pd.DataFrame()
			return z1_out, z2_out, labels, pat_info_df
		elif self.pooler == 'trivial':
			# Generate positive pairs using the same embeding with a bit of gaussian noise injected
			# Used for sanity check on task difficulty
			# add in patient_id saving and index saving for downstream error analysis
			z1_out = torch.tensor([]).to(self.device)
			z2_out = torch.tensor([]).to(self.device)
			labels = []
			for i, e in enumerate(embeds):
				idx = np.random.randint(0,e.shape[0]-1)
				z1_out = torch.concat((z1_out, torch.reshape(e[idx], (1,1,embeds.shape[-1]))), 0)
				z2_out = torch.concat((z2_out, torch.reshape(e[idx] + (0.01**0.5)*torch.randn_like(e[idx]), (1,1,embeds.shape[-1]))), 0)
				labels.append(1)
				neg_embed_idx = [j for j in list(np.arange(len(embeds))) if j != i]
				if len(neg_embed_idx) > 5:
					neg_embed_idx = np.random.choice(neg_embed_idx, 1)
				for j in neg_embed_idx:				
					neg_embeds = embeds[j]
					z1_out = torch.concat((z1_out, torch.reshape(e[idx], (1,1,embeds.shape[-1]))), 0)
					neg_idx = np.random.randint(0, neg_embeds.shape[0])
					z2_out = torch.concat((z2_out, torch.reshape(neg_embeds[neg_idx], (1,1,embeds.shape[-1]))), 0)
					labels.append(0)
			
			labels = torch.tensor(labels).float().to(self.device)

			return z1_out, z2_out, labels

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
		self.criterion = nn.BCEWithLogitsLoss()
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
		if self.pooler.pooler == 'diff_pat' or self.pooler.pooler == 'trivial':
			z1_target_embeds, z2_target_embeds, labels, pat_df = self.pooler(z1_embeds, pids=batch['pid'])
		else:
			z1_target_embeds, z2_target_embeds, labels, pat_df = self.pooler(z1_embeds, z2_embeds, rand_day_indices, pids=batch['pid'])
		
		if z1_target_embeds is None:
			return None
		# Reshape pooled embeds to BATCH_SIZE X 2 X EMBEDDING_SIZE
		# First column is z1, second column is z2
		if len(z1_target_embeds) == 0:
			pooled_embeds = torch.zeros((1,2,800)).to(self.device)
		else:
			pooled_embeds = torch.concat((z1_target_embeds, z2_target_embeds), axis=1)

			# pooled_embeds = pooled_embeds.view(len(z1_target_embeds), 2, pooled_embeds.size(-1))
		
		pooled_embeds = self.linear(pooled_embeds)
		z1, z2 = pooled_embeds[:,0], pooled_embeds[:,1]
		
		# Get cosine similarity
		cos_sim = self.sim(z1, z2)

		# Compute loss
		outputs['loss'] = self.criterion(cos_sim,labels)
		outputs['preds'] = cos_sim
		outputs['labels'] = labels
		outputs['pat_df'] = pat_df
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
        
def finetune_model(args, model, dataset, lr, clmbr_save_path, clmbr_model_path):
	"""
	Finetune CLMBR model using linear layer.
	"""
	model.train()
	config = model.clmbr_model.config
	optimizer = optim.Adam([p for n, p in model.named_parameters() if p.requires_grad], lr=lr)
	early_stop = EarlyStopping(args.patience)
	best_val_loss = 9999999
	best_epoch = 0
	
	for e in range(args.epochs):
		
		os.makedirs(f'{clmbr_save_path}/{e}',exist_ok=True)
		
		model.train()
		pat_info_df = pd.DataFrame()
		model_train_loss_df = pd.DataFrame()
		model_val_loss_df = pd.DataFrame()
		train_loss = []
		train_preds = []
		train_lbls = []
		with DataLoader(dataset, model.config['num_first'], is_val=False, batch_size=model.config["batch_size"], device=args.device) as train_loader:
			for batch in tqdm(train_loader):
				# Skip batches that only consist of one patient - otherwise no negative samples are generated
				if len(batch['pid']) < 2:
					continue
				else:
					optimizer.zero_grad()
					outputs = model(batch)
					if outputs is not None:
						df = outputs['pat_df']
						df['epoch'] = e
						df['phase'] = 'train'
					
					pat_info_df = pd.concat((pat_info_df,df))
					if outputs is None:
						continue
					train_preds.extend(list(outputs['preds'].detach().clone().cpu().numpy()))
					train_lbls.extend(list(outputs['labels'].detach().clone().cpu().numpy()))
					loss = outputs["loss"]

					loss.backward()
					optimizer.step()
					train_loss.append(loss.item())
		print('Training loss:',  np.sum(train_loss))
		df = pd.DataFrame({'loss':train_loss})
		df['epoch'] = e
		df.to_csv(f'{clmbr_save_path}/{e}/train_loss.csv')
		
		# evaluate on validation set
		val_preds, val_lbls, val_losses, df = evaluate_model(args, model, dataset, e)
		df['epoch'] = e
		df['phase'] = 'val'
		pat_info_df = pd.concat((pat_info_df,df))
		scaled_val_loss = np.sum(val_losses)*model.temp
		df = pd.DataFrame({'loss':val_losses})
		df['epoch'] = e
		df.to_csv(f'{clmbr_save_path}/{e}/val_loss.csv')
		
		# Save train and val model predictions/labels
		df = pd.DataFrame({'epoch':e,'preds':train_preds,'labels':train_lbls})
		df.to_csv(f'{clmbr_save_path}/{e}/train_preds.csv', index=False)
		df = pd.DataFrame({'epoch':e,'preds':val_preds,'labels':val_lbls})
		df.to_csv(f'{clmbr_save_path}/{e}/val_preds.csv', index=False)
		
		#save current epoch model
		os.makedirs(f'{clmbr_save_path}/{e}',exist_ok=True)
		torch.save(model.clmbr_model.state_dict(), os.path.join(clmbr_save_path,f'{e}/best'))
		shutil.copyfile(f'{clmbr_model_path}/info.json', f'{clmbr_save_path}/{e}/info.json')
		pat_info_df.to_csv(f'{clmbr_save_path}/{e}/pat_info.csv', index=False)
		with open(f'{clmbr_save_path}/{e}/config.json', 'w') as f:
			json.dump(config,f)			
		
		#save model as best model if condition met
		if scaled_val_loss < best_val_loss:
			best_val_loss = scaled_val_loss
			best_epoch = e
			best_model = copy.deepcopy(model.clmbr_model)
		
		# Trigger early stopping if model hasn't improved for awhile
		if early_stop(scaled_val_loss):
			print(f'Early stopping at epoch {e}')
			break
	
	# save best epoch for debugging 
	with open(f'{clmbr_save_path}/best_epoch.txt', 'w') as f:
		f.write(f'{best_epoch}')
		
	return best_model, best_val_loss, best_epoch

def evaluate_model(args, model, dataset, e):
	model.eval()
	
	criterion = nn.CrossEntropyLoss()
	
	preds = []
	lbls = []
	losses = []
	pat_info_df = pd.DataFrame()
	with torch.no_grad():
		with DataLoader(dataset, model.config['num_first'], is_val=True, batch_size=model.config['batch_size'], seed=args.seed, device=args.device) as eval_loader:
			for batch in tqdm(eval_loader):
				if len(batch['pid']) < 2:
					continue
				outputs = model(batch)
				if outputs is None:
					continue
				loss = outputs["loss"]
				losses.append(loss.item())
				preds.extend(list(outputs['preds'].cpu().numpy()))
				lbls.extend(list(outputs['labels'].cpu().numpy()))
				df = outputs['pat_df']

				pat_info_df = pd.concat((pat_info_df,df))
	print('Validation loss:',  np.sum(losses))
	return preds, lbls, losses, pat_info_df

def finetune(args, cl_hp, clmbr_model_path, dataset):
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
	clmbr_model, val_loss, best_epoch = finetune_model(args, model, dataset, float(cl_hp['lr']), clmbr_save_path, clmbr_model_path)
	clmbr_model.freeze()

	# Save model and save info and config to new model directory for downstream evaluation
	torch.save(clmbr_model.state_dict(), os.path.join(clmbr_save_path,'best'))
	shutil.copyfile(f'{clmbr_model_path}/info.json', f'{clmbr_save_path}/info.json')
	with open(f'{clmbr_save_path}/config.json', 'w') as f:
		json.dump(config,f)
	
	return clmbr_model, val_loss, best_epoch

if __name__ == '__main__':
    
	args = parser.parse_args()

	torch.manual_seed(args.seed)
	
	clmbr_hp = list(
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
	
	cl_grid = list(
		ParameterGrid(
			yaml.load(
				open(
					f"{os.path.join(args.hparams_fpath,f'{args.pooler}')}.yml",
					'r'
				),
				Loader=yaml.FullLoader
			)
		)
	)

	print(args.pooler)
	clmbr_model_path = f'{args.pt_model_path}/{args.encoder}_sz_{clmbr_hp["size"]}_do_{clmbr_hp["dropout"]}_cd_{clmbr_hp["code_dropout"]}_dd_{clmbr_hp["day_dropout"]}_lr_{clmbr_hp["lr"]}_l2_{clmbr_hp["l2"]}'
	print(clmbr_model_path)
	best_val_loss = 9999999
	best_params = None
	
	train_data, val_data = load_data(args, clmbr_hp)

	dataset = PatientTimelineDataset(args.extract_path + '/extract.db', 
									 args.extract_path + '/ontology.db', 
									 f'{clmbr_model_path}/info.json', 
									 train_data, 
									 val_data )
	
	for j, cl_hp in enumerate(cl_grid):
		print('finetuning model with params: ', cl_hp)
		best_ft_path = f"{args.model_path}/{args.encoder}_sz_{clmbr_hp['size']}_do_{clmbr_hp['dropout']}_cd_{clmbr_hp['code_dropout']}_dd_{clmbr_hp['day_dropout']}_lr_{clmbr_hp['lr']}_l2_{clmbr_hp['l2']}/best_{cl_hp['pool']}"
		clmbr_save_path = f"{args.model_path}/{args.encoder}_sz_{clmbr_hp['size']}_do_{clmbr_hp['dropout']}_cd_{clmbr_hp['code_dropout']}_dd_{clmbr_hp['day_dropout']}_lr_{clmbr_hp['lr']}_l2_{clmbr_hp['l2']}/bs_{cl_hp['batch_size']}_lr_{cl_hp['lr']}_temp_{cl_hp['temp']}_pool_{cl_hp['pool']}"
		print(clmbr_save_path)
	
		os.makedirs(f"{clmbr_save_path}",exist_ok=True)
	
		
		os.makedirs(f"{best_ft_path}",exist_ok=True)
		
		model, val_loss, best_epoch = finetune(args, cl_hp, clmbr_model_path, dataset)
		
		if val_loss < best_val_loss:
			print('Saving as best finetuned model...')
			best_val_loss = val_loss
			best_params = cl_hp

			torch.save(model.state_dict(), os.path.join(best_ft_path,'best'))
			shutil.copyfile(f'{clmbr_model_path}/info.json', f'{best_ft_path}/info.json')
			with open(f'{best_ft_path}/config.json', 'w') as f:
				json.dump(model.config,f)
			with open(f"{best_ft_path}/hyperparams.yml", 'w') as file: 
				yaml.dump(best_params,file)
			with open(f'{best_ft_path}/best_epoch.txt', 'w') as f:
				f.write(f'{best_epoch}')
        
    