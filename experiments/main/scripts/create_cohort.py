import configargparse as argparse
import pandas as pd
import os
from prediction_utils.cohorts.admissions.cohort import BQAdmissionRollupCohort
from prediction_utils.cohorts.admissions.cohort import BQAdmissionOutcomeCohort
from prediction_utils.cohorts.admissions.cohort import BQFilterInpatientCohort
from prediction_utils.util import patient_split_cv

parser = argparse.ArgumentParser()

parser.add_argument(
    "--dataset", type=str, default="starr_omop_cdm5_deid_20210723"
)
parser.add_argument("--rs_dataset", type=str, default="temp_dataset")
parser.add_argument("--et_dataset", type=str, default="jlemmon_explore")
parser.add_argument("--limit", type=int, default=0)
parser.add_argument("--gcloud_project", type=str, default="som-nero-nigam-starr")
parser.add_argument("--dataset_project", type=str, default="som-nero-nigam-starr")
parser.add_argument(
    "--rs_dataset_project", type=str, default="som-nero-nigam-starr"
)
parser.add_argument("--cohort_name", type=str, default="admission_rollup_temp")
parser.add_argument(
    "--cohort_name_labeled", type=str, default="admission_rollup_labeled_temp"
)
parser.add_argument(
    "--cohort_name_filtered", type=str, default="admission_rollup_filtered_temp"
)
parser.add_argument(
    "--has_birth_datetime", dest="has_birth_datetime", action="store_true"
)
parser.add_argument(
    "--no_has_birth_datetime", dest="has_birth_datetime", action="store_false"
)
parser.add_argument(
    "--data_path",
    type=str,
    default="/local-scratch/nigam/projects/jlemmon/cl-clmbr/experiments/main/data/",
)
parser.add_argument(
    "--min_stay_hour",
    type=int,
    default=0,
)
parser.add_argument(
    "--filter_query",
    type=str,
    default="",
)
parser.add_argument(
    "--google_application_credentials",
    type=str,
    default=os.path.expanduser("~/.config/gcloud/application_default_credentials.json"),
)
parser.set_defaults(has_birth_datetime=True)

def append_extra_task(args, df, task):
	# credentials, _ = google.auth.default()
	query = f'SELECT * FROM {args.dataset_project}.{args.et_dataset}.{task}_cohort'
	et_cohort_df = pd.read_gbq(query, dialect='standard')
	temp_df = df.merge(et_cohort_df, on='person_id', how='inner')
	temp_df['start_date'] = pd.to_datetime(temp_df['start_date'], format='%Y-%m-%d')
	temp_df = temp_df.loc[temp_df.start_date > temp_df.admit_date]
	pat_ids = list(temp_df['person_id'])
	df[task] = 0
	df[task] = df[task].mask(df.person_id.isin(pat_ids), 1)
	return df


if __name__ == "__main__":
	
	extra_tasks = ['sudden_cardiac_death', 'stroke', 'bladder_cancer', 'breast_cancer', 'acute_renal_failure', 'acute_myocardial_infarction', 'diabetic_ketoacidosis', 'edema', 'hyperkylemia', 'renal_cancer', 'revascularization']

	args = parser.parse_args()
	cohort = BQAdmissionRollupCohort(**args.__dict__)
	print(cohort.get_create_query())
	cohort.create_cohort_table()

	cohort_labeled = BQAdmissionOutcomeCohort(**args.__dict__)
	print(cohort_labeled.get_create_query())
	cohort_labeled.create_cohort_table()

	cohort_filtered = BQFilterInpatientCohort(**args.__dict__)
	print(cohort_labeled.get_create_query())
	cohort_filtered.create_cohort_table()
	cohort_df = pd.read_gbq(
		"""
			SELECT *
			FROM `{rs_dataset_project}.{rs_dataset}.{cohort_name_filtered}`
		""".format(
			**args.__dict__
		),
		dialect='standard',
	)
	cohort_df = patient_split_cv(
		cohort_df, patient_col="person_id", test_frac=0.1, nfold=10, seed=657
	)
	for et in extra_tasks:
		cohort_df = append_extra_task(args, cohort_df, et)
	cohort_df['death_date'] = pd.to_datetime(cohort_df['death_date'])
	print(cohort_df.dtypes)
	
	cohort_path = os.path.join(args.data_path, "cohort")
	os.makedirs(cohort_path, exist_ok=True)
	cohort_df.to_parquet(
		os.path.join(cohort_path, "cohort.parquet"), engine="pyarrow", index=False,
	)