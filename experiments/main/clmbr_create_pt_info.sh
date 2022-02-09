source activate /local-scratch/nigam/envs/jlemmon/conl


EXTRACT_DIR='/local-scratch/nigam/projects/jlemmon/cl-clmbr/experiments/main/data/extracts/20210723'
SAVE_DIR='/local-scratch/nigam/projects/jlemmon/cl-clmbr/experiments/main/artifacts/models/clmbr/pretrained/info'
EXCLUDE_FILE_DIR='/local-scratch/nigam/projects/jlemmon/cl-clmbr/experiments/main/data/held_out_patients/excluded_patient_ids.txt'

clmbr_create_info $EXTRACT_DIR $SAVE_DIR '2015-12-31' '2016-07-01' --excluded_patient_file $EXCLUDE_FILE_DIR