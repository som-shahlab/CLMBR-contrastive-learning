import os
import argparse
import yaml
import shutil
import pdb
import torch
import joblib

import pandas as pd
import numpy as np

from itertools import zip_longest
from subprocess import (run, Popen)
from prediction_utils.util import str2bool
from ehr_ml.clmbr import convert_patient_data
from sklearn.model_selection import ParameterGrid

parser = argparse.ArgumentParser(
    description='Train baseline CLMBR model'
)

parser.add_argument(
    '--min_patient_count', 
    type=str,
    default="100",
)

parser.add_argument(
    '--extracts_fpath', 
    type=str,
    default="/local-scratch/nigam/projects/jlemmon/cl-clmbr/experiments/main/data/extracts/20210723",
)

parser.add_argument(
    '--cohort_fpath',
    type=str,
    default="/local-scratch/nigam/projects/jlemmon/cl-clmbr/cohort",
)

parser.add_argument(
    '--infos_fpath',
    type=str,
    default="/local-scratch/nigam/projects/jlemmon/cl-clmbr/experiments/main/artifacts/models/clmbr/pretrained/info"
)

parser.add_argument(
    '--models_fpath',
    type=str,
    default="/local-scratch/nigam/projects/jlemmon/cl-clmbr/experiments/main/artifacts/models/clmbr/pretrained/models"
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
    '--excluded_patient_list',
    type=str,
    default="/local-scratch/nigam/projects/jlemmon/cl-clmbr/experiments/main/data/held_out_patients/excluded_patient_ids.txt"
)

parser.add_argument(
    '--hparams_fpath',
    type=str,
    default="/local-scratch/nigam/projects/jlemmon/cl-clmbr/experiments/main/hyperparams/"
)

parser.add_argument(
    '--encoder',
    type=str,
    default='gru',
    help='Encoder type: GRU/Transformer',
)

parser.add_argument(
    '--overwrite',
    type=str2bool,
    default='false'
)

parser.add_argument(
    '--n_gpu',
    type=int,
    default=1
)

parser.add_argument(
    '--n_jobs',
    type=int,
    default=8
)

parser.add_argument(
    '--gpu_num_start',
    type=int,
    default=3
)

parser.add_argument(
	'--seed',
	type=int,
	default=44
)

parser.add_argument(
	'--device',
	type=str,
	default='cuda:0'
)

if __name__ == "__main__":
    
	args = parser.parse_args()
	
	# threads
	torch.set_num_threads(1)
	joblib.Parallel(n_jobs=1)

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
    
	# create info
	info_dir=f'{args.infos_fpath}'
	train_end_date=args.train_end_date
	val_end_date=args.val_end_date
	
	
	if args.overwrite and os.path.exists(info_dir):
		shutil.rmtree(info_dir, ignore_errors=True)

	run([
		'clmbr_create_info',
		f"{args.extracts_fpath}",
		f"{info_dir}",
		f"{train_end_date}",
		f"{val_end_date}",
		"--train_start_date", f"{args.train_start_date}",
		"--val_start_date", f"{args.val_start_date}",
		"--min_patient_count", args.min_patient_count,
		"--excluded_patient_file", args.excluded_patient_list,
		"--seed", f'{args.seed}'
	])
	
	processes=[]
    
    # collect args
	for i,hparams in enumerate(grid):
        
		model_dir=f'{args.models_fpath}/{args.encoder}_sz_{hparams["size"]}_do_{hparams["dropout"]}_cd_{hparams["code_dropout"]}_dd_{hparams["day_dropout"]}_lr_{hparams["lr"]}_l2_{hparams["l2"]}'

		if args.overwrite and os.path.exists(model_dir):
			shutil.rmtree(model_dir, ignore_errors=True)
			os.makedirs(model_dir, exist_ok=True)
        
		torch.manual_seed(args.seed)
		
		p_args = [
			'clmbr_train_model',
			model_dir,
			info_dir,
			'--lr', f"{hparams['lr']}",
			'--encoder_type', f"{hparams['encoder_type']}",
			'--size', f"{hparams['size']}",
			'--dropout', f"{hparams['dropout']}",
			'--code_dropout', f"{hparams['code_dropout']}",
			'--day_dropout', f"{hparams['day_dropout']}",
			'--batch_size', f"{hparams['batch_size']}",
			'--epochs', f"{hparams['epochs']}",
			'--l2', f"{hparams['l2']}",
			'--warmup_epochs', f"{hparams['warmup_epochs']}",
			'--device', f'{args.device}',
		]
        
		processes.append(p_args)

    # group processes 
	processes = [
		(
			Popen(
				p,
				env=dict(os.environ, CUDA_VISIBLE_DEVICES = str(i%args.n_gpu+args.gpu_num_start))
			) for i,p in enumerate(processes)
		)
	] * args.n_jobs

    # submit n_jobs jobs at a time
	for sub_p in zip_longest(*processes): 
		for p in filter(None, sub_p):
			p.wait()