"""
Use CLMBR labeler classes to label dataset for downstream classification tasks.
"""

import argparse
import os
import pickle

import pandas as pd
import numpy as np

from utils_prediction.utils import str2bool

import ehr_ml.timeline
import ehr_ml.labeler
import ehr_ml.clmbr


parser = argparse.ArgumentParser(description='Arguments for dataset labelling.')

parser.add_argument(
    '--task',
    type=str,
    required=True,
    help='Name of task to label. Accepted tasks are:\nmortality\nlong admission\nHbA1c\nreadmission'
)

parser.add_argument(
    '--cohort_path',
    type=str,
    default='/local-scratch/nigam/projects/jlemmon/cl-clmbr/Experiments/main/data/cohorts',
    help='Path to dataset directory.'
)

parser.add_Argument(
    '--extract_path',
    type=str,
    default='/local-scratch/nigam/projects/jlemmon/cl-clmbr/Experiments/main/data/extracts/20210723'
)

parser.add_argument(
    '--seed',
    type=int,
    default=44,
    help='Random seed value.'
)


if __name__ == "__main__":
    
    args = parser.parse_args()
    np.random.seed(args.seed)
    
    labeler_save_path = args.labeler_path + '/' + args.task
    
    timelines = ehr_ml.timeline.TimelineReader(os.path.join(args.extract_path, 'extract.db'))
    
    if args.task == 'mortality':
        labeler = ehr_ml.labeler.RandomSelectionLabeler(ehr_ml.labeler.InpatientMortalityLabeler(timelines), random_seed = RANDOM_SEED)
    elif args.task == 'long admission':
        labeler = ehr_ml.labeler.RandomSelectionLabeler(ehr_ml.labeler.LongAdmissionLabeler(timelines), random_seed = RANDOM_SEED)
    elif args.task == 'HbA1c':
        labeler = ehr_ml.labeler.RandomSelectionLabeler(ehr_ml.labeler.HighHbA1cLabeler(timelines), random_seed = RANDOM_SEED)
    elif args.task == 'readmission':
        labeler = ehr_ml.labeler.RandomSelectionLabeler(ehr_ml.labeler.InpatientReadmissionLabeler(timelines), random_seed = RANDOM_SEED)
    elif args.task == 'icu':
        #need to find ICU labeler code
        pass
    
    ehr_ml.labeler.SavedLabeler.save(labeler, timelines, labeler_save_path)
    with open(labeler_save_path) as f:
        saved_labeler = ehr_ml.labeler.SavedLabeler(f)