import os
import json
import argparse
import yaml
import pickle
from datetime import datetime

import ehr_ml.timeline
import ehr_ml.ontology
import ehr_ml.index
import ehr_ml.labeler
import ehr_ml.clmbr
from ehr_ml.clmbr import convert_patient_data

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterGrid
from scipy.sparse import csr_matrix as csr
from sklearn.linear_model import LogisticRegression as lr

import torch
import torch.nn as nn
import torch.nn.functional as F

from itertools import zip_longest
from subprocess import (run, Popen)
from prediction_utils.pytorch_utils.metrics import StandardEvaluator

parser = argparse.ArgumentParser()

parser.add_argument(
    '--model_path',
    type=str,
    default='/local-scratch/nigam/projects/jlemmon/cl-clmbr/experiments/main/artifacts/models/clmbr',
    help='Base path for the trained end-to-end model.'
)

parser.add_argument(
    '--result_path',
    type=str,
    default='/local-scratch/nigam/projects/jlemmon/cl-clmbr/experiments/main/artifacts/results',
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
    '--hparams_fpath',
    type=str,
    default="/local-scratch/nigam/projects/jlemmon/cl-clmbr/experiments/main/hyperparams/"
)

parser.add_argument(
    '--train_start_date',
    type=str,
    default='2008-01-01',
    help='Start date of training ids.'
)

parser.add_argument(
    '--train_end_date',
    type=str,
    default='2016-12-31',
    help='End date of training ids.'
)

parser.add_argument(
    '--val_start_date',
    type=str,
    default='2008-01-01',
    help='Start date of validation ids.'
)

parser.add_argument(
    '--val_end_date',
    type=str,
    default='2016-12-31',
    help='End date of validation ids.'
)

parser.add_argument(
    '--test_start_date',
    type=str,
    default='2017-01-01',
    help='Start date of test ids.'
)

parser.add_argument(
    '--test_end_date',
    type=str,
    default='2021-12-31',
    help='End date of test ids.'
)

parser.add_argument(
    '--cohort_dtype',
    type=str,
    default='parquet',
    help='Data type for cohort file.'
)

parser.add_argument(
	'--model',
	type=str,
	default='lr'
)

parser.add_argument(
    '--size',
    type=int,
    default=800,
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
    default=0.001,
    help='Learning rate for training.'
)

parser.add_argument(
    '--encoder',
    type=str,
    default='gru',
    help='Underlying encoder architecture for CLMBR. [gru|transformer|lstm]'
)

parser.add_argument(
    "--n_jobs",
    type=int,
    default=4,
    help="number of threads"
)

parser.add_argument(
    "--n_searches",
    type=int,
    default=100,
    help="number of random searches to conduct for hparam search"
)

parser.add_argument(
    '--models_path',
    type=str,
    default='/local-scratch/nigam/projects/jlemmon/cl-clmbr/experiments/main/artifacts/models',
    help='Base path for models.'
)

parser.add_argument(
    "--n_boot",
    type=int,
    default=10000,
    help="number of bootstrap iterations for evaluation"
)

parser.add_argument(
	'--device',
	type=str,
	default='cuda:0'
)

parser.add_argument(
	'--seed',
	type=int,
	default=44
)

def get_params(args):
    param_grid = yaml.load(
        open(
            os.path.join(
                args.hparams_fpath,
                f'{args.model}.yml'
            ), 'r'
        ),
        Loader = yaml.FullLoader
    )
    
    param_grid = list(ParameterGrid(param_grid))
    np.random.shuffle(param_grid)
    
    if args.n_searches < len(param_grid):
        param_grid = param_grid[:args.n_searches]
    
    return param_grid

if __name__ == '__main__':
    
	args = parser.parse_args()

	np.random.seed(args.seed)

	tasks = ['hospital_mortality', 'LOS_7', 'icu_admission', 'readmission_30', 'sudden_cardiac_death', 'stroke', 'bladder_cancer', 'breast_cancer', 'acute_renal_failure', 'acute_myocardial_infarction', 'diabetic_ketoacidosis', 'edema', 'hyperkylemia', 'renal_cancer', 'revascularization']

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
    
	for i, hparams in enumerate(grid):

		clmbr_model_path = f'{args.model_path}/pretrained/models/{args.encoder}_sz_{hparams["size"]}_do_{hparams["dropout"]}_cd_{hparams["code_dropout"]}_dd_{hparams["day_dropout"]}_lr_{hparams["lr"]}_l2_{hparams["l2"]}'

		if args.cohort_dtype == 'parquet':
			dataset = pd.read_parquet(os.path.join(args.cohort_fpath, "cohort_split.parquet"))
		else:
			dataset = pd.read_csv(os.path.join(args.cohort_fpath, "cohort_split.csv"))
		
		train_start_date = pd.to_datetime(args.train_start_date)
		train_end_date = pd.to_datetime(args.train_end_date)
		val_start_date = pd.to_datetime(args.val_start_date)
		val_end_date = pd.to_datetime(args.val_end_date)
		test_start_date = pd.to_datetime(args.test_start_date)
		test_end_date = pd.to_datetime(args.test_end_date)

		dataset = dataset.assign(date = pd.to_datetime(dataset['admit_date']).dt.date)

		ehr_ml_patient_ids = {}
		prediction_ids = {}
		day_indices = {}
		labels = {}
		features = {}

		clmbr_model = ehr_ml.clmbr.CLMBR.from_pretrained(f'{clmbr_model_path}', device=args.device)
		for task in tasks:

			ehr_ml_patient_ids[task] = {}
			prediction_ids[task] = {}
			day_indices[task] = {}
			labels[task] = {}
			features[task] = {}

			if task == 'readmission':
				index_year = 'discharge_year'
			else:
				index_year = 'admission_year'

			for fold in ['train', 'val', 'test']:
				print(f'Featurizing task {task} fold {fold}')

				if fold == 'train':
					df = dataset.query(f"{task}_fold_id!=['test','val','ignore']")
					print('train end:', train_end_date)
					mask = ((df['date'] >= train_start_date) & (df['date'] <= train_end_date))
				elif fold == 'val':
					df = dataset.query(f"{task}_fold_id==['val']")
					print('val start:', val_start_date, 'val end:', val_end_date)
					mask = ((df['date'] >= val_start_date) & (df['date'] <= val_end_date))
				else:
					df = dataset.query(f"{task}_fold_id==['test']")
					print('test start:', test_start_date, 'test end:', test_end_date)
					mask = ((df['date'] >= test_start_date) & (df['date'] <= test_end_date))
				df = df.loc[mask].reset_index()
				print(min(df['date']))
				print(max(df['date']))
				ehr_ml_patient_ids[task][fold], day_indices[task][fold] = convert_patient_data(args.extract_path, df['person_id'], 
																							   df['admit_date'].dt.date if task!='readmission_30' else df['discharge_date'].dt.date)
				labels[task][fold] = df[task]
				prediction_ids[task][fold]=df['prediction_id']


				assert(
					len(ehr_ml_patient_ids[task][fold]) ==
					len(labels[task][fold]) == 
					len(prediction_ids[task][fold])
				)

				features[task][fold] = clmbr_model.featurize_patients(args.extract_path, np.array(ehr_ml_patient_ids[task][fold]), np.array(day_indices[task][fold]))

				print('Saving artifacts...')

				task_path = f'{args.labelled_fpath}/{task}/pretrained/{args.encoder}_sz_{hparams["size"]}_do_{hparams["dropout"]}_cd_{hparams["code_dropout"]}_dd_{hparams["day_dropout"]}_lr_{hparams["lr"]}_l2_{hparams["l2"]}'
				print(task_path)
				if not os.path.exists(task_path):
					os.makedirs(task_path)
				df_ehr_pat_ids = pd.DataFrame(ehr_ml_patient_ids[task][fold])
				df_ehr_pat_ids.to_csv(f'{task_path}/ehr_ml_patient_ids_{fold}.csv', index=False)

				df_prediction_ids = pd.DataFrame(prediction_ids[task][fold])
				df_prediction_ids.to_csv(f'{task_path}/prediction_ids_{fold}.csv', index=False)

				df_day_inds = pd.DataFrame(day_indices[task][fold])
				df_day_inds.to_csv(f'{task_path}/day_indices_{fold}.csv', index=False)

				df_labels = pd.DataFrame(labels[task][fold])
				df_labels.to_csv(f'{task_path}/labels_{fold}.csv', index=False)

				# df_features = pd.DataFrame(features[task][fold])
				# df_features.to_csv(f'{task_path}/features_{fold}.csv', index=False)
				with open(f'{task_path}/features_{fold}.pkl', 'wb') as f:
					pickle.dump(features[task][fold], f)
				
                
            
        
    