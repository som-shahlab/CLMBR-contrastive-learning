import json
import torch
import os
import argparse

from ehr_ml.clmbr import CLMBR
from ehr_ml.clmbr.dataset import DataLoader
from ehr_ml.clmbr.opt import OpenAIAdam
from ehr_ml.clmbr import PatientTimelineDataset


parser = argparse.ArgumentParser(description='Arguments for CMBR pretraining.')

parser.add_argument(
    '--model_path',
    type=str,
    default='/local-scratch/nigam/projects/jlemmon/cl-clmbr/experiments/main/artifacts/models/clmbr/pretrained/models',
    help='Base path for the pretrained model.'
)

parser.add_argument(
    '--info_path',
    type=str,
    default='/local-scratch/nigam/projects/jlemmon/cl-clmbr/experiments/main/artifacts/models/clmbr/pretrained/info',
    help='Base path for the pretraining info.json'
)

parser.add_argument(
    '--extract_path',
    type=str,
    default='/local-scratch/nigam/projects/jlemmon/cl-clmbr/experiments/main/data/extracts/20210723',
    help='Base path for the extracted database.'
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
    default=0.01,
    help='Learning rate for training.'
)

parser.add_argument(
    '--nn_type',
    type=str,
    default='gru',
    help='Underlying neural network architecture for CLMBR. [gru|transformer|lstm]'
)

parser.add_argument(
    '--epochs',
    type=int,
    default=50,
    help='Number of training epochs.'
)

parser.add_argument(
    '--warmup',
    type=int,
    default=2,
    help='Number of warmup epochs.'
)

parser.add_argument(
    '--batch_size',
    type=int,
    default=500,
    help='Size of batch samples.'
)

parser.add_argument(
    '--eval_batch_size',
    type=int,
    default=2000,
    help='Size of evaluation batch samples.'
)

parser.add_argument(
    '--l2',
    type=float,
    default=0.01,
    help='Regularization constant for L2 LR model.'
)
parser.add_argument(
    '--device',
    type=str,
    default='cpu',
    help='UName of device to run training on'
)

# Functions

def get_batches(args):
    dataset = PatientTimelineDataset(os.path.join(args.extract_path, "extract.db"), 
                                 os.path.join(args.extract_path, "ontology.db"),
                                 os.path.join(args.info_path, "info.json"))
    batches = DataLoader(dataset, threshold=config["num_first"], is_val=False, batch_size=args.batch_size)


if __name__ == '__main__':
    
    args = parser.parse_args()
        
    batches = get_batches(args)