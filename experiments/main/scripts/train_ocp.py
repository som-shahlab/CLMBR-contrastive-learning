import os
import argparse
import yaml
import shutil
import pdb
import torch
import joblib

import pandas as pd
import numpy as np

import ocp

from itertools import zip_longest
from subprocess import (run, Popen)
from prediction_utils.util import str2bool
from ehr_ml.clmbr import convert_patient_data
from sklearn.model_selection import ParameterGrid

parser = argparse.ArgumentParser(
    description='Train CLMBR model'
)

parser.add_argument(
    '--hparams_fpath',
    type=str,
    default="/local-scratch/nigam/projects/jlemmon/cl-clmbr/experiments/main/hyperparams"
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
    default=2
)

parser.add_argument(
    '--n_jobs',
    type=int,
    default=8
)

parser.add_argument(
    '--gpu_num_start',
    type=int,
    default=4
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
					f"{os.path.join(args.hparams_fpath,'ocp')}.yml",
					'r'
				),
				Loader=yaml.FullLoader
			)
		)
	)

	processes=[]

	# collect args
	for i,hparams in enumerate(grid):

		p_args = [
			'python',
			'ocp.py',
			'--multi_gpu', '1',
			'--lr', f"{hparams['lr']}",
			'--pooler', f"{hparams['pool']}",
			'--encoder', f"{hparams['encoder_type']}",
			'--size', f"{hparams['size']}",
			'--dropout', f"{hparams['dropout']}",
			'--epochs', f"{hparams['epochs']}",
			'--l2', f"{hparams['l2']}",
			'--warmup_epochs', f"{hparams['warmup_epochs']}",
			'--device', 'cuda:0',
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