import os
import numpy as np
import pandas as pd
from ehr_ml.clmbr import PatientTimelineDataset
from ehr_ml.clmbr.dataset import DataLoader
from ehr_ml.timeline import TimelineReader
from ehr_ml.clmbr import convert_patient_data
import google.auth

credentials, project_id = google.auth.default()
rs_dataset_project = 'som-nero-nigam-starr'
rs_dataset = 'jlemmon_explore'
rs_table = 'clmbr_admission_rollup'
clmbr_model_path = '/local-scratch/nigam/projects/jlemmon/cl-clmbr/experiments/main/artifacts/models/clmbr/pretrained/models/gru_sz_800_do_0.1_cd_0_dd_0_lr_0.001_l2_0.01'
cohort_fpath = '/local-scratch/nigam/projects/jlemmon/cl-clmbr/experiments/main/data/'
extract_path = os.path.join('/local-scratch/nigam/projects/jlemmon/cl-clmbr/experiments/main/data/extracts/20210723')
labelled_fpath = '/local-scratch/nigam/projects/jlemmon/cl-clmbr/experiments/main/data/labelled_data/hospital_mortality/pretrained/gru_sz_800_do_0.1_cd_0_dd_0_lr_0.001_l2_0.01'

query = f'SELECT * FROM {rs_dataset_project}.{rs_dataset}.{rs_table}'

adm_df = pd.read_gbq(query, project_id=rs_dataset_project, dialect='standard', credentials=credentials)

filter_df = adm_df[adm_df.groupby('person_id').person_id.transform('count') >1]
ehr_val_pids, ehr_val_start_days = convert_patient_data(extract_path, filter_df['person_id'], filter_df['admit_date'].dt.date)
ehr_val_pids, ehr_val_end_days = convert_patient_data(extract_path, filter_df['person_id'], filter_df['discharge_date'].dt.date)
ocp_ids = pd.DataFrame({'pid':list(filter_df['person_id']),'ehr_id':ehr_val_pids, 'admit_date':list(filter_df['admit_date']), 'discharge_date':list(filter_df['discharge_date']),'start_day_idx':ehr_val_start_days, 'end_day_idx':ehr_val_end_days})

save_dir = "/local-scratch/nigam/projects/jlemmon/cl-clmbr/experiments/main/data/ocp_patients"

os.makedirs(save_dir, exist_ok=True)

ocp_ids.to_csv(save_dir + '/patients.csv', index=False)
