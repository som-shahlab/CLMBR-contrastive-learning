import os
import json
import argparse
from datetime import datetime

import ehr_ml.timeline
import ehr_ml.ontology
import ehr_ml.index
import ehr_ml.labeler
import ehr_ml.clmbr
from ehr_ml.clmbr import PatientTimelineDataset

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    default='/local-scratch/nigam/projects/jlemmon/cl-clmbr/experiments/main/artifacts/models/clmbr/baseline/models',
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


class LinearCLMBRClassifier(nn.Module):
    def __init__(self, clmbr_model, num_classes, device=None):
        super().__init__()
        self.clmbr_model = clmbr_model
        self.linear = nn.Linear(clmbr_model.config["size"], num_classes)
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def forward(self, batch):
        embedding = self.clmbr_model.timeline_model(batch["rnn"])

        label_indices, label_values = batch["label"]

        flat_embeddings = embedding.view((-1, embedding.shape[-1]))

        target_embeddings = F.embedding(label_indices, flat_embeddings)

        return self.linear(target_embeddings), label_values

    
def load_data(args):
    train_pids = pd.from_csv(f'{args.labelled_fpath}/{args.task}/ehr_ml_patient_ids_train')
    val_pids = pd.from_csv(f'{args.labelled_fpath}/{args.task}/ehr_ml_patient_ids_val')
    test_pids = pd.from_csv(f'{args.labelled_fpath}/{args.task}/ehr_ml_patient_ids_test')
    
    train_days = pd.from_csv(f'{args.labelled_fpath}/{args.task}/day_indices_train')
    val_days = pd.from_csv(f'{args.labelled_fpath}/{args.task}/day_indices_val')
    test_days = pd.from_csv(f'{args.labelled_fpath}/{args.task}/day_indices_test')
    
    train_labels = pd.from_csv(f'{args.labelled_fpath}/{args.task}/labels_train')
    val_labels = pd.from_csv(f'{args.labelled_fpath}/{args.task}/labels_val')
    test_labels = pd.from_csv(f'{args.labelled_fpath}/{args.task}/labels_test')
    
    train_data = (train_labels, train_pids, train_days)
    val_data = (val_labels, val_pids, val_days)
    test_data = (test_labels, test_pids, test_days)
    
    return train_data, val_data, test_data


if __name__ == '__main__':
    
    args = parser.parse_args()
    
    clmbr_model_path = f"{args.pt_model_path}/{args.nn_type}_{args.size}_{args.dropout}"
    
    if args.cohort_dtype == 'parquet':
        dataset = pd.read_parquet(os.path.join(args.cohort_fpath, "cohort.parquet"))
    else:
        dataset = pd.read_csv(os.path.join(args.cohort_fpath, "cohort_split.csv"))
    
    
    dataset = dataset.assign(date = pd.to_datetime(dataset['admit_date']).dt.date)
    print(dataset.columns)
    
    

    train_end_date = datetime.strptime(args.train_end_date, '%Y-%m-%d')
    print(train_end_date)
    val_end_date = datetime.strptime(args.val_end_date, '%Y-%m-%d')
    train = dataset.query("fold_id!=['val','test','1'] and date<=@train_end_date")

    print(train)
    
    val = dataset.query("fold_id=[1'] and date>=train_end_date and date<=@val_end_date")

    print(val)
    
    clmbr_model = enr_ml.clmbr.CLMBR.from_pretrained(clmbr_model_path)
    
        
    