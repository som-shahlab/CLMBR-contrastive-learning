import os
import json
import argparse
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

import torch
import torch.nn as nn
import torch.nn.functional as F

parser = argparse.ArgumentParser()

parser.add_argument(
    '--model_path',
    type=str,
    default='/local-scratch/nigam/projects/jlemmon/cl-clmbr/experiments/main/artifacts/models/clmbr/',
    help='Base path for the trained end-to-end model.'
)

parser.add_argument(
    '--clmbr_type',
    type=str,
    default='pretrained'
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
    '--test_start_date',
    type=str,
    default='2016-09-01',
    help='Start date of test ids.'
)

parser.add_argument(
    '--test_end_date',
    type=str,
    default='2017-09-01',
    help='End date of test ids.'
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
    default=0.001,
    help='Learning rate for training.'
)

parser.add_argument(
    '--encoder',
    type=str,
    default='gru',
    help='Underlying encoder architecture for CLMBR. [gru|transformer|lstm]'
)

if __name__ == '__main__':
    
    args = parser.parse_args()
    
	np.random.seed(args.seed)
	
    tasks = ['hospital_mortality', 'LOS_7', 'icu_admission', 'readmission_30']
    
    clmbr_model_path = f"{args.model_path}/{args.clmbr_type}/models/{args.encoder}_sz_{args.size}_do_{args.dropout}_lr_{args.lr}_l2_{args.l2}"
    
    if args.cohort_dtype == 'parquet':
        dataset = pd.read_parquet(os.path.join(args.cohort_fpath, "cohort_split.parquet"))
    else:
        dataset = pd.read_csv(os.path.join(args.cohort_fpath, "cohort_split.csv"))
    
    train_end_date = pd.to_datetime(args.train_end_date)
    val_end_date = pd.to_datetime(args.val_end_date)
    test_start_date = pd.to_datetime(args.test_start_date)
    test_end_date = pd.to_datetime(args.test_end_date)
    
    dataset = dataset.assign(date = pd.to_datetime(dataset['admit_date']).dt.date)
    
    ehr_ml_patient_ids = {}
    prediction_ids = {}
    day_indices = {}
    labels = {}
    features = {}
    
    clmbr_model = ehr_ml.clmbr.CLMBR.from_pretrained(f'{clmbr_model_path}')
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
                mask = (df['date'] <= train_end_date)
            elif fold == 'val':
                df = dataset.query(f"{task}_fold_id==['val']")
                mask = (df['date'] >= train_end_date) & (df['date'] <= val_end_date)
            else:
                df = dataset.query(f"{task}_fold_id==['test']")
                mask = (df['date'] >= test_start_date) & (df['date'] <= test_end_date)
            df = df.loc[mask].reset_index()

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
            
			task_path = f"{args.labelled_fpath}/{args.clmbr_type}/{args.encoder}_sz_{args.size}_do_{args.dropout}_lr_{args.lr}_l2_{args.l2}/{task}"
			
            if not os.path.exists(f'task_path'):
                os.makedirs(f'task_path')
            df_ehr_pat_ids = pd.DataFrame(ehr_ml_patient_ids[task][fold])
            df_ehr_pat_ids.to_csv(f'task_path/ehr_ml_patient_ids_{fold}.csv', index=False)
            
            df_prediction_ids = pd.DataFrame(prediction_ids[task][fold])
            df_prediction_ids.to_csv(f'task_path/prediction_ids_{fold}.csv', index=False)
            
            df_day_inds = pd.DataFrame(day_indices[task][fold])
            df_day_inds.to_csv(f'task_path/day_indices_{fold}.csv', index=False)
            
            df_labels = pd.DataFrame(labels[task][fold])
            df_labels.to_csv(f'task_path/labels_{fold}.csv', index=False)
            
            df_features = pd.DataFrame(features[task][fold])
            df_features.to_csv(f'task_path/features_{fold}.csv', index=False)

                                      
                
            
        
    