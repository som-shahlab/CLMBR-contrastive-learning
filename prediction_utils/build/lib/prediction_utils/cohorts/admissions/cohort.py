from prediction_utils.cohorts.cohort import BQCohort


class BQAdmissionRollupCohort(BQCohort):
	"""
	Cohort that defines an admission rollup
	"""

	def get_base_query(self, format_query=True):
		query = """ (
		SELECT * FROM (
			/*
			SELECT 
				t1.person_id, 
				visit_detail_concept_id as visit_concept_id, 
				visit_detail_start_datetime as visit_start_datetime, 
				visit_end_datetime
			FROM {dataset_project}.{dataset}.visit_detail t1
			INNER JOIN {dataset_project}.{dataset}.person as t2
				ON t1.person_id = t2.person_id
			INNER JOIN {dataset_project}.{dataset}.visit_occurrence t3
				ON t1.visit_occurrence_id = t3.visit_occurrence_id
			WHERE
				visit_detail_concept_id in (9201, 262)
				AND visit_end_datetime > visit_detail_start_datetime
				AND visit_end_datetime is not NULL
				AND visit_detail_start_datetime is not NULL

			UNION DISTINCT
			*/
			SELECT 
				t1.person_id, 
				visit_concept_id, 
				visit_start_datetime, 
				visit_end_datetime
			FROM {dataset_project}.{dataset}.visit_occurrence t1
			INNER JOIN {dataset_project}.{dataset}.person as t2
				ON t1.person_id = t2.person_id
			WHERE
				visit_concept_id in (9201, 262)
				AND visit_end_datetime > visit_start_datetime
				AND visit_end_datetime is not NULL
				AND visit_start_datetime is not NULL
		)
		{where_str}
		{limit_str}
		)
		"""
		if not format_query:
			return query
		else:
			return query.format_map(self.config_dict)

	def get_transform_query(self, format_query=True):
		query = """
			WITH visits AS (
			  SELECT *
			  FROM {base_query}
			),
			visits_melt AS (
			  SELECT person_id, visit_start_datetime AS endpoint_date, 1 as endpoint_type
			  FROM visits
			  UNION ALL
			  SELECT person_id, visit_end_datetime AS endpoint_date, -1 as endpoint_type
			  FROM visits
			),
			counts1 AS (
			  SELECT *, COUNT(*) * endpoint_type as count
			  FROM visits_melt
			  GROUP BY person_id, endpoint_date, endpoint_type
			),
			counts2 AS (
			  SELECT person_id, endpoint_date, SUM(count) as count
			  FROM counts1
			  GROUP BY person_id, endpoint_date
			),
			counts3 AS (
			  SELECT person_id, endpoint_date,
				  SUM(count) OVER(PARTITION BY person_id ORDER BY endpoint_date) as count
			  FROM counts2
			),
			cum_counts AS (
			  SELECT *, ROW_NUMBER() OVER(PARTITION BY person_id ORDER BY endpoint_date) as row_number
			  FROM counts3
			),
			discharge_times AS (
			  SELECT person_id, endpoint_date, 'discharge_date' as endpoint_type, row_number
			  FROM cum_counts
			  WHERE count = 0
			),
			discharge_times_row_shifted AS (
			  SELECT person_id, (row_number + 1) as row_number
			  FROM discharge_times
			),
			first_admit_times AS (
			  SELECT person_id, endpoint_date, 'admit_date' as endpoint_type
			  FROM cum_counts
			  WHERE row_number = 1
			),
			other_admit_times AS (
			  SELECT t1.person_id, endpoint_date, 'admit_date' as endpoint_type
			  FROM cum_counts t1
			  INNER JOIN discharge_times_row_shifted AS t2
			  ON t1.person_id=t2.person_id AND t1.row_number=t2.row_number
			),
			aggregated_endpoints AS (
			  SELECT person_id, endpoint_date, endpoint_type
			  FROM discharge_times
			  UNION ALL
			  SELECT person_id, endpoint_date, endpoint_type
			  FROM first_admit_times
			  UNION ALL
			  SELECT person_id, endpoint_date, endpoint_type
			  FROM other_admit_times
			),
			result_long AS (
			  SELECT *, ROW_NUMBER() OVER(PARTITION BY person_id, endpoint_type ORDER BY endpoint_date) as row_number
			  FROM aggregated_endpoints
			),
			discharge_times_final AS (
				SELECT person_id, endpoint_date as discharge_date, row_number
				FROM result_long
				WHERE endpoint_type = 'discharge_date'
			),
			admit_times_final AS (
				SELECT person_id, endpoint_date as admit_date, row_number
				FROM result_long
				WHERE endpoint_type = 'admit_date'
			),
			result AS (
				SELECT t1.person_id, admit_date, discharge_date, t1.row_number
				FROM admit_times_final t1
				INNER JOIN discharge_times_final as t2
				ON t1.person_id=t2.person_id AND t1.row_number=t2.row_number
			)
			SELECT person_id, admit_date, discharge_date
			FROM result
			ORDER BY person_id, row_number
		"""

		if not format_query:
			return query
		else:
			return query.format_map(
				{**self.config_dict, **{"base_query": self.get_base_query()}}
			)

	def get_create_query(self, format_query=True):

		query = """ 
			CREATE OR REPLACE TABLE {rs_dataset_project}.{rs_dataset}.{cohort_name} AS
			{query}
		"""

		if not format_query:
			return query
		else:
			return query.format_map(
				{**self.config_dict, **{"query": self.get_transform_query()}}
			)


class BQAdmissionOutcomeCohort(BQCohort):
	"""
	Cohort that defines
	"""

	def get_config_dict(self, **kwargs):
		config_dict = super().get_config_dict(**kwargs)
		if config_dict["has_birth_datetime"]:
			config_dict[
				"age_in_years"
			] = "DATE_DIFF(CAST(admit_date AS DATE), CAST(birth_datetime AS DATE), YEAR)"
		else:
			config_dict[
				"age_in_years"
			] = "EXTRACT(YEAR FROM admit_date) - year_of_birth"
		return config_dict

	def get_defaults(self):
		config_dict = super().get_defaults()

		config_dict["has_birth_datetime"] = True
		config_dict["cohort_name_labeled"] = "temp_cohort_labeled"
		return config_dict

	def get_base_query(self, format_query=True):

		query = "{rs_dataset_project}.{rs_dataset}.{cohort_name}"
		if not format_query:
			return query
		else:
			return query.format_map(self.config_dict)

	def get_create_query(self, format_query=True):
		query = """ 
			CREATE OR REPLACE TABLE {rs_dataset_project}.{rs_dataset}.{cohort_name_labeled} AS
			{query}
		"""

		if not format_query:
			return query
		else:
			return query.format_map(
				{**self.config_dict, **{"query": self.get_transform_query()}}
			)

	def get_transform_query(self, format_query=True):
		query = """
			WITH mortality_labels AS (
				{mortality_query}
			),
			month_mortality_labels AS (
				{month_mortality_query}
			),
			LOS_labels AS (
				{los_query}
			), 
			readmission_labels AS (
				{readmission_query}
			), 
			icu_labels AS (
				{icu_query}
			),
			age_labels AS (
				{age_query}
			),
			aki_labels AS (
				{aki_query}
			),
			np_labels AS (
				{np_query}
			),
			hg_labels AS (
				{hg_query}
			),
			demographics AS (
				{demographics_query}
			),
			race_eth_raw AS (
				{race_eth_raw_query}
			),
			end_of_ad_day AS (
				{end_of_ad_day_query}
			),
			cohort_with_labels AS (
				SELECT *
				FROM {base_query}
				LEFT JOIN end_of_ad_day USING (person_id, admit_date, discharge_date)
				LEFT JOIN mortality_labels USING (person_id, admit_date, discharge_date)
				LEFT JOIN month_mortality_labels USING (person_id, admit_date, discharge_date)
				LEFT JOIN LOS_labels USING (person_id, admit_date, discharge_date)
				LEFT JOIN readmission_labels USING (person_id, admit_date, discharge_date)
				LEFT JOIN icu_labels USING (person_id, admit_date, discharge_date)
				LEFT JOIN age_labels USING (person_id, admit_date, discharge_date)
				LEFT JOIN aki_labels USING (person_id, admit_date, discharge_date)
				LEFT JOIN hg_labels USING (person_id, admit_date, discharge_date)
				LEFT JOIN np_labels USING (person_id, admit_date, discharge_date)
				LEFT JOIN demographics USING (person_id)
				LEFT JOIN race_eth_raw USING (person_id)
			)
			-- append intersectional demographics definitions
			SELECT *, 
				CONCAT('race_eth:', race_eth, '-gender:', gender_concept_name) AS race_eth_gender,
				CONCAT('race_eth:', race_eth, '-age_group:', age_group) AS race_eth_age_group,
				CONCAT('race_eth:', race_eth, '-gender:', gender_concept_name, '-age_group:', age_group) AS race_eth_gender_age_group,
				CONCAT('race_eth_raw:', race_eth_raw, '-gender:', gender_concept_name) AS race_eth_raw_gender,
				CONCAT('race_eth_raw:', race_eth_raw, '-age_group:', age_group) AS race_eth_raw_age_group,
				CONCAT('race_eth_raw:', race_eth_raw, '-gender:', gender_concept_name, '-age_group:', age_group) AS race_eth_raw_gender_age_group
			FROM cohort_with_labels
		"""

		if not format_query:
			return query
		else:
			return query.format(
				base_query=self.get_base_query(), **self.get_label_query_dict()
			)

	def get_label_query_dict(self):
		query_dict = {
			"mortality_query": self.get_hospital_mortality_query(),
			"month_mortality_query": self.get_month_mortality_query(),
			"los_query": self.get_los_query(),
			"readmission_query": self.get_readmission_query(),
			"icu_query": self.get_icu_query(),
			"age_query": self.get_age_query(),
			"aki_query": self.get_aki_query(),
			"np_query": self.get_np_query(),
			"hg_query": self.get_hg_query(),
			"demographics_query": self.get_demographics_query(),
			"race_eth_raw_query": self.get_race_eth_raw_query(),
			"end_of_ad_day_query":self.get_end_of_ad_day_query(),
		}
		base_query = self.get_base_query()
		return {
			key: value.format_map({**{"base_query": base_query}, **self.config_dict})
			for key, value in query_dict.items()
		}

	def get_end_of_ad_day_query(self):
		return """
			WITH temp as (
				SELECT 
					person_id,admit_date,discharge_date,
					datetime_add( 
						datetime_trunc(
							datetime_add(admit_date, INTERVAL 1 day),
							day
						),
						INTERVAL -1 minute
					) as admit_date_midnight,
					datetime_add( 
						datetime_trunc(
							datetime_add(discharge_date, INTERVAL 1 day),
							day
						),
						INTERVAL -1 minute
					) as discharge_date_midnight 
				FROM {base_query} t1
			)
			SELECT t1.*, admit_date_midnight, discharge_date_midnight
			FROM {base_query} t1
			LEFT JOIN temp USING (person_id, admit_date, discharge_date)
		"""

	def get_hospital_mortality_query(self):
		return """
			WITH temp AS (
				SELECT t1.person_id, admit_date, discharge_date, death_date,
				CASE
					WHEN death_date BETWEEN admit_date AND discharge_date THEN 1
					ELSE 0
				END as hospital_mortality
				FROM {base_query} t1
				RIGHT JOIN {dataset_project}.{dataset}.death AS t2
					ON t1.person_id = t2.person_id
			)
			SELECT t1.*, IFNULL(hospital_mortality, 0) as hospital_mortality, death_date
			FROM {base_query} t1
			LEFT JOIN temp USING (person_id, admit_date, discharge_date)
		"""

	def get_month_mortality_query(self):
		return """
			WITH temp AS (
				SELECT t1.person_id, admit_date, discharge_date, death_date,
				CASE
					WHEN DATE_TRUNC(death_date, MONTH) = DATE_TRUNC(discharge_date, MONTH) THEN 1
					ELSE 0
				END as month_mortality
				FROM {base_query} t1
				RIGHT JOIN {dataset_project}.{dataset}.death AS t2
					ON t1.person_id = t2.person_id
			)
			SELECT t1.*, IFNULL(month_mortality, 0) as month_mortality
			FROM {base_query} t1
			LEFT JOIN temp USING (person_id, admit_date, discharge_date)
		"""

	def get_los_query(self):
		return """
			WITH temp AS (
				SELECT *,
				DATE_DIFF(t1.discharge_date, t1.admit_date, DAY) AS LOS_days
				FROM {base_query} t1
			)
			SELECT *, 
			CAST(LOS_days >= 7 AS INT64) as LOS_7
			FROM temp
		"""

	def get_readmission_query(self):
		return """
			WITH temp AS (
				SELECT *, ROW_NUMBER() OVER(PARTITION BY person_id ORDER BY admit_date) as row_number
				FROM {base_query}
			),
			temp_shifted AS ( -- row shift by one
				SELECT person_id, admit_date, discharge_date, row_number - 1 as row_number
				FROM temp
			),
			temp_readmission_window AS ( --compare discharges from temp to admits from temp_shifted
				SELECT t1.person_id, 
					t1.discharge_date, 
					t2.admit_date, 
					DATE_DIFF(t2.admit_date, t1.discharge_date, DAY) AS readmission_window,
					t1.row_number
				FROM temp as t1
				INNER JOIN temp_shifted as t2
				ON t1.person_id = t2.person_id AND t1.row_number = t2.row_number
			),
			result AS (
				SELECT t1.person_id, t1.admit_date, t1.discharge_date, t2.readmission_window, 
				CASE 
					WHEN readmission_window BETWEEN 0 AND 30 THEN 1
					ELSE 0
				END as readmission_30
				FROM temp as t1
				INNER JOIN temp_readmission_window as t2
				on t1.person_id = t2.person_id AND t1.row_number = t2.row_number
			)
			SELECT t1.*, IFNULL(readmission_30, 0) as readmission_30, readmission_window
			FROM {base_query} t1
			LEFT JOIN result USING (person_id, admit_date, discharge_date)
		"""

	def get_icu_query(self):
		return """
			WITH icu1 AS
			(
			 SELECT
			  t1.*,
			  detail.visit_detail_start_datetime AS icu_start_datetime,
			FROM
			  {base_query} t1
			LEFT JOIN
			  {dataset_project}.{dataset}.visit_detail detail
			ON
			  detail.person_id = t1.person_id
			  AND visit_detail_start_datetime BETWEEN t1.admit_date AND t1.discharge_date
			  AND visit_detail_start_datetime > DATETIME_ADD(t1.admit_date, INTERVAL {min_stay_hour} hour)
			  AND visit_detail_source_value IN
			   ('J4|J4|J4|',
				'J2|J2|J2|',
				'K4|K4|K4|',
				'M4|M4|M4|',
				'L4|L4|L4|',
				'ACA6 ICU|ACA6ICU|ACA6ICU|',
				'E2-ICU|E2|E2-ICU|Intensive Care',
				'VCP CCU 2|VCPC2|VCP CCU 2|Critical Care Medicine',
				'VCP CCU 1|VCPC1|VCP CCU 1|Critical Care Medicine',
				'D2ICU-SURGE|D2ICU|D2ICU|Intensive Care',
				'2 NORTH|2NCVICU|Cardiovascular Intensive Care|',
				'CVICU 220|CVICU220|Cardiovascular Intensive Care 220|',
				'CVICU 320|CVICU320|Cardiovascular Intensive Care 320|',
				'CVICU|CVICU|Cardiovascular Intensive Care|',
				'E29-ICU|NICU|E29-ICU|Intensive Care',
				'NICU 260|NICU260|Neonatal Intensive Care|Neonatology',
				'NICU 270|NICU270|Neonatal Intensive Care|Neonatology',
				'PICU 320|PICU320|Pediatric Intensive Care 320|',
				'PICU 420|PICU420|Pediatric Intensive Care 420|',
				'PICU|PICU|Pediatric Intensive Care|',
				'VCP NICU|VCPNICU|VCP NICU|Neonatology'
				)
			),
			icu2 AS
			(
			  SELECT
			   icu1.*,
			   RANK() OVER(
				   PARTITION BY icu1.person_id, icu1.admit_date, icu1.discharge_date
				   ORDER BY icu1.icu_start_datetime ASC
			   ) rank_
			  FROM
			   icu1
			),
			icu3 AS
			(
			  SELECT
			   icu2.*,
			  FROM
			   icu2
			  WHERE rank_ = 1
			)
			SELECT t1.*, CASE WHEN icu_start_datetime IS NULL THEN 0 ELSE 1 END as icu_admission, icu_start_datetime
			FROM {base_query} t1
			LEFT JOIN icu3 USING (person_id, admit_date, discharge_date)
		"""

	def get_age_query(self):
		return """
			WITH temp AS (
				SELECT t1.person_id, admit_date, discharge_date,
				{age_in_years} AS age_in_years,
				--DATE_DIFF(CAST(admit_date AS DATE), CAST(birth_datetime AS DATE), YEAR) AS age_in_years
				FROM {base_query} t1
				INNER JOIN {dataset_project}.{dataset}.person t2
				ON t1.person_id = t2.person_id
			)
			SELECT person_id, admit_date, discharge_date,
			age_in_years,
			CASE 
				WHEN age_in_years >= 18.0 and age_in_years < 30.0 THEN '[18-30)'
				WHEN age_in_years >= 30.0 and age_in_years < 45.0 THEN '[30-45)'
				WHEN age_in_years >= 45.0 and age_in_years < 55.0 THEN '[45-55)'
				WHEN age_in_years >= 55.0 and age_in_years < 65.0 THEN '[55-65)'
				WHEN age_in_years >= 65.0 and age_in_years < 75.0 THEN '[65-75)'
				WHEN age_in_years >= 75.0 and age_in_years < 91.0 THEN '[75-91)'
				ELSE '<18'
			END as age_group
			FROM temp
		"""

	def get_demographics_query(self):
		return """
			SELECT DISTINCT t1.person_id,
				CASE
					WHEN t5.concept_name = "Hispanic or Latino" THEN "Hispanic or Latino"
					WHEN t4.concept_name = "Other, Hispanic" THEN "Hispanic or Latino"
					WHEN t4.concept_name in (
						"Native Hawaiian or Other Pacific Islander",
						"American Indian or Alaska Native",
						"No matching concept",
						"Other, non-Hispanic",
						"Race and Ethnicity Unknown",
						"Unknown",
						"Declines to State", 
						"Patient Refused"
					) THEN "Other"
					ELSE t4.concept_name
				END as race_eth,
				t3.concept_name as gender_concept_name
				FROM {base_query} t1
				INNER JOIN {dataset_project}.{dataset}.person AS t2
					ON t1.person_id = t2.person_id
				INNER JOIN {dataset_project}.{dataset}.concept as t3
					ON t2.gender_concept_id = t3.concept_id
				INNER JOIN {dataset_project}.{dataset}.concept as t4
					ON t2.race_concept_id = t4.concept_id
				INNER JOIN {dataset_project}.{dataset}.concept as t5
					ON t2.ethnicity_concept_id = t5.concept_id
		"""

	def get_race_eth_raw_query(self):
		"""
		This query does not collapse racial categories
		"""
		return """
			SELECT DISTINCT t1.person_id,
				CASE
					WHEN t5.concept_name = "Hispanic or Latino" THEN "Hispanic or Latino"
					WHEN t4.concept_name = "No matching concept" THEN "Other"
					ELSE t4.concept_name
				END as race_eth_raw
				FROM {base_query} t1
				INNER JOIN {dataset_project}.{dataset}.person AS t2
					ON t1.person_id = t2.person_id
				INNER JOIN {dataset_project}.{dataset}.concept as t4
					ON t2.race_concept_id = t4.concept_id
				INNER JOIN {dataset_project}.{dataset}.concept as t5
					ON t2.ethnicity_concept_id = t5.concept_id
		"""

	def get_aki_query(self):
		return """
			WITH measurement_concepts as (
				SELECT 
					c.concept_id, concept_name
				FROM `{dataset_project}.{dataset}.concept` c 
				WHERE c.concept_id in (37029387,4013964,2212294,3051825)

				UNION DISTINCT 

				SELECT 
					c.concept_id, concept_name
				FROM `{dataset_project}.{dataset}.concept` c
				INNER JOIN `{dataset_project}.{dataset}.concept_ancestor` ca 
					ON c.concept_id = ca.descendant_concept_id
					AND ca.ancestor_concept_id in (37029387,4013964,2212294,3051825)
					AND c.invalid_reason is null 
			),
			all_measurements AS
			(
			SELECT
			  t1.*, m.measurement_datetime
			  -- convert micromol/liter (8749) to miligram/deciliter (8840)
			  ,case
				  when m.unit_concept_id = 8749 then m.value_as_number * 0.0113122
				  when m.unit_concept_id = 8837 then m.value_as_number * 0.001
				  else m.value_as_number
				end as value_as_number
			FROM {base_query} t1
			LEFT JOIN {dataset_project}.{dataset}.measurement m
				ON t1.person_id = m.person_id
				AND m.unit_concept_id IN (
					8749,  -- umol/l (x0.0113122 to get mg/dl)
					8840,  -- mg/dl
					8837   -- ug/dl (x0.001 to get mg/dl)
				  )
			INNER JOIN measurement_concepts mc 
				ON m.measurement_concept_id = mc.concept_id
			),
			-- GET BASELINE CREATININE MEASUREMENTS (MIN MEASUREMENT BETWEEN ADMISSION TIME AND 3 MONTHS PRIOR)
			base_3month as (
				SELECT person_id,admit_date, discharge_date
					,MIN(m.value_as_number) as value_as_number
				FROM all_measurements m 
				WHERE measurement_datetime >= date_add(admit_date, INTERVAL -90 day)
					AND measurement_datetime < admit_date 
				GROUP BY person_id, admit_date, discharge_date
			),
			base_by_age_norm as (
				SELECT m.person_id,admit_date,discharge_date
					,CASE 
						WHEN DATE_DIFF(admit_date, birth_datetime, DAY) BETWEEN 0 AND 14 THEN 0.92 
						WHEN DATE_DIFF(admit_date, birth_datetime, DAY) BETWEEN 15 AND (2*365-1) THEN 0.36
						WHEN DATE_DIFF(admit_date, birth_datetime, DAY) BETWEEN (2*365) AND (5*365-1) THEN 0.43
						WHEN DATE_DIFF(admit_date, birth_datetime, DAY) BETWEEN (5*365) AND (12*365-1) THEN 0.61
						WHEN DATE_DIFF(admit_date, birth_datetime, DAY) BETWEEN (12*365) AND (15*365-1) THEN 0.81
						WHEN DATE_DIFF(admit_date, birth_datetime, DAY) BETWEEN (15*365) AND (19*365-1) THEN 0.84
						WHEN DATE_DIFF(admit_date, birth_datetime, DAY) >= (19*365) AND gender_concept_id=8532 THEN 1.1
						WHEN DATE_DIFF(admit_date, birth_datetime, DAY) >= (19*365) AND gender_concept_id<>8532 THEN 1.2
						ELSE NULL
					END AS value_as_number
				FROM (SELECT DISTINCT person_id, admit_date, discharge_date FROM all_measurements) m
				LEFT JOIN `{dataset_project}.{dataset}.person` person
					ON m.person_id=person.person_id
			),
			base_measurements as (
				SELECT b_age.person_id, b_age.admit_date, b_age.discharge_date 
					,COALESCE(b_3month.value_as_number, b_age.value_as_number) as value_as_number
				FROM base_by_age_norm b_age 
				LEFT JOIN base_3month b_3month 
					ON b_age.person_id=b_3month.person_id 
					AND b_age.admit_date=b_3month.admit_date
					AND b_age.discharge_date=b_3month.discharge_date
			),
			max_measurements as (
				SELECT m.person_id, admit_date, discharge_date
					,MAX(m.value_as_number) as aki_max_creatinine
				FROM all_measurements m 
				WHERE measurement_datetime > admit_date 
					AND measurement_datetime <= discharge_date 
				GROUP BY person_id, admit_date, discharge_date
			),
			all_abnormals_during_admission_aki1 as
			(
				SELECT m.person_id, m.admit_date, m.discharge_date 
					,m.value_as_number 
					,m.measurement_datetime
					,ROW_NUMBER() OVER(
						PARTITION BY m.person_id, m.admit_date, m.discharge_date
						ORDER BY m.measurement_datetime ASC
					) rn
				FROM all_measurements m
				LEFT JOIN base_measurements b USING (person_id, admit_date, discharge_date)
				WHERE measurement_datetime > admit_date 
					AND measurement_datetime <= discharge_date 
					AND (
						m.value_as_number/b.value_as_number >= 1.5
						OR m.value_as_number-b.value_as_number >= 0.3
					)
			),
			first_abnormal_during_admission_aki1 as
			(
				SELECT person_id, admit_date, discharge_date
					,value_as_number as aki_creatinine
					,measurement_datetime as aki_creatinine_time
				FROM all_abnormals_during_admission_aki1
				WHERE rn = 1
			),
			aki1 as (
				SELECT b.person_id, b.admit_date, b.discharge_date
					,b.value_as_number as aki_base_creatinine
					,aki_max_creatinine
					,aki_creatinine
					,aki_creatinine_time
					,CASE 
						WHEN aki_creatinine IS NOT NULL THEN 1
						ELSE 0 
					END AS aki_label
				FROM base_measurements b
				LEFT JOIN max_measurements m USING (person_id, admit_date, discharge_date)
				LEFT JOIN first_abnormal_during_admission_aki1 ab USING (person_id, admit_date, discharge_date)
			),
			all_abnormals_during_admission_aki2 as
			(
				SELECT m.person_id, m.admit_date, m.discharge_date 
					,m.value_as_number 
					,m.measurement_datetime
					,ROW_NUMBER() OVER(
						PARTITION BY m.person_id, m.admit_date, m.discharge_date
						ORDER BY m.measurement_datetime ASC
					) rn
				FROM all_measurements m
				LEFT JOIN base_measurements b USING (person_id, admit_date, discharge_date)
				WHERE measurement_datetime > admit_date 
					AND measurement_datetime <= discharge_date 
					AND (
						m.value_as_number/b.value_as_number >= 2.0
					)
			),
			first_abnormal_during_admission_aki2 as
			(
				SELECT person_id, admit_date, discharge_date
					,value_as_number as aki_creatinine
					,measurement_datetime as aki_creatinine_time
				FROM all_abnormals_during_admission_aki2
				WHERE rn = 1
			),
			aki2 as (
				SELECT b.person_id, b.admit_date, b.discharge_date
					,b.value_as_number as aki_base_creatinine
					,aki_creatinine
					,aki_creatinine_time
					,CASE 
						WHEN aki_creatinine IS NOT NULL THEN 1
						ELSE 0 
					END AS aki_label
				FROM base_measurements b
				LEFT JOIN first_abnormal_during_admission_aki2 ab USING (person_id, admit_date, discharge_date)
			)
			SELECT t1.*
				,aki1.aki_base_creatinine, aki1.aki_max_creatinine
				,aki1.aki_creatinine as aki1_creatinine, aki1.aki_creatinine_time as aki1_creatinine_time
				,CASE 
					WHEN aki1.aki_label IS NULL THEN 0 
					ELSE aki1.aki_label
				END AS aki1_label
				,aki2.aki_creatinine as aki2_creatinine, aki2.aki_creatinine_time as aki2_creatinine_time
				,CASE 
					WHEN aki2.aki_label IS NULL THEN 0 
					ELSE aki2.aki_label
				END AS aki2_label
			FROM {base_query} t1
			LEFT JOIN aki1 USING (person_id, admit_date, discharge_date)
			LEFT JOIN aki2 USING (person_id, admit_date, discharge_date)
		"""


	def get_hg_query(self):
		return """
			WITH measurement_concepts as 
			(
				SELECT 
					c.concept_id, concept_name
				FROM `{dataset_project}.{dataset}.concept` c 
				WHERE c.concept_id in (4144235, 1002597)

				UNION DISTINCT

				SELECT 
					c.concept_id, concept_name
				FROM `{dataset_project}.{dataset}.concept` c
				INNER JOIN `{dataset_project}.{dataset}.concept_ancestor` ca 
					ON c.concept_id = ca.descendant_concept_id
					AND ca.ancestor_concept_id in (4144235, 1002597)
					AND c.invalid_reason is null 
			),
			all_measurements AS
			(
				SELECT
				  t1.*, m.measurement_datetime
				  ,CASE
					  WHEN m.unit_concept_id = 8753 THEN m.value_as_number * 18 
					  ELSE m.value_as_number
				  END AS value_as_number
				FROM {base_query} t1
				LEFT JOIN `{dataset_project}.{dataset}.measurement` m 
				  on t1.person_id = m.person_id
				  AND m.unit_concept_id IN (
					  8840, -- mg/dL
					  9028, -- mg/dL calculated
					  8753 -- mmol/L (x 18 to get mg/dl)
				  )
				INNER JOIN measurement_concepts mc 
				  ON m.measurement_concept_id = mc.concept_id
			),
			min_measurements as 
			(
				SELECT person_id
					,admit_date, discharge_date
					,min(value_as_number) as hg_min_glucose
				FROM all_measurements
				WHERE measurement_datetime > admit_date 
					AND measurement_datetime <= discharge_date 
				GROUP BY person_id, admit_date, discharge_date
			),
			all_abnormals_during_admission as
			(
				SELECT person_id, admit_date, discharge_date 
					,value_as_number 
					,measurement_datetime
					,ROW_NUMBER() OVER(
						PARTITION BY person_id, admit_date, discharge_date
						ORDER BY measurement_datetime ASC
					) rn
				FROM all_measurements
				WHERE measurement_datetime > admit_date 
					AND measurement_datetime <= discharge_date 
					AND value_as_number <= 60
			),
			first_abnormal_during_admission as
			(
				SELECT person_id, admit_date, discharge_date
					,value_as_number as hg_glucose
					,measurement_datetime as hg_glucose_time
				FROM all_abnormals_during_admission
				WHERE rn = 1
			),
			hg AS 
			(
				SELECT m.person_id, m.admit_date, m.discharge_date
					,hg_min_glucose
					,hg_glucose
					,hg_glucose_time
					,CASE WHEN hg_glucose IS NOT NULL THEN 1 ELSE 0 END AS hg_label
				FROM min_measurements m 
				LEFT JOIN first_abnormal_during_admission ab USING (person_id, admit_date, discharge_date)
			)
			SELECT t1.*, hg_min_glucose, hg_glucose, hg_glucose_time
				,CASE 
					WHEN hg.hg_label is NULL then 0
					ELSE hg.hg_label 
				END AS hg_label
			FROM {base_query} t1
			LEFT JOIN hg USING (person_id, admit_date, discharge_date)
		"""

	def get_np_query(self):
		return """
			-- WBC (LEUKOCYTES)
			WITH wbc_concepts as 
			(
				-- Use 3010813
				-- 3000905 has unusual distribution
				SELECT 
					c.concept_id, 'wbc' as concept_name
				FROM `{dataset_project}.{dataset}.concept` c
				WHERE concept_id = 3010813       
			),
			all_wbc as 
			(
			  SELECT t1.*
				  ,m.measurement_datetime
				  ,m.value_as_number as wbc
			  FROM {base_query} t1
			  LEFT JOIN `{dataset_project}.{dataset}.measurement` m 
				  ON t1.person_id = m.person_id
				  AND m.unit_concept_id IN (
					8848,  -- 1000/uL
					8961  -- 1000/mm^3, equivalent to 8848
				  )
			  INNER JOIN wbc_concepts mc 
				  on m.measurement_concept_id = mc.concept_id
			),
			-- BANDS
			bands_concepts as 
			(
				-- 3035839 band form /100 leukocytes (%)
				-- 3018199 band form neutrophils in blood (count) 
				SELECT 
					c.concept_id, concept_name
				FROM `{dataset_project}.{dataset}.concept` c
				WHERE concept_id in (3035839, 3018199)
			),
			all_bands as
			(
				SELECT t1.*
					,m.measurement_datetime 
					,CASE 
						WHEN m.measurement_concept_id = 3035839 AND m.value_as_number<=100
						  THEN m.value_as_number / 100 * wbc.wbc
						WHEN m.measurement_concept_id = 3018199 AND m.unit_concept_id = 8784 
						  THEN m.value_as_number / 1000
						ELSE NULL
					END AS bands_count
				FROM {base_query} t1
				LEFT JOIN `{dataset_project}.{dataset}.measurement` m 
					ON t1.person_id = m.person_id
				INNER JOIN bands_concepts mc 
					ON m.measurement_concept_id = mc.concept_id
				LEFT JOIN all_wbc wbc 
					ON m.person_id = wbc.person_id
					AND m.measurement_datetime = wbc.measurement_datetime
			),
			-- NEUTROPHILS
			neutrophils_concepts as 
			(
				SELECT 
					c.concept_id, concept_name
				FROM `{dataset_project}.{dataset}.concept` c
				WHERE concept_id in (37045722, 37049637)

				UNION DISTINCT 

				SELECT 
					c.concept_id, concept_name
				FROM `{dataset_project}.{dataset}.concept` c
				INNER JOIN `{dataset_project}.{dataset}.concept_ancestor` ca 
					ON c.concept_id = ca.descendant_concept_id
					AND ca.ancestor_concept_id in (37045722, 37049637)
					AND c.invalid_reason is null 
			),
			all_neutrophils as 
			(
				SELECT t1.*
				  ,m.measurement_datetime 
				  ,CASE 
					  -- neutrophils /100 leukocytes
					  WHEN m.unit_concept_id = 8554 AND m.value_as_number<=100
						  THEN m.value_as_number / 100 * wbc.wbc
					  WHEN m.unit_concept_id = 8554 AND m.value_as_number>100
						  THEN NULL
					  WHEN m.unit_concept_id = 8784  
						  THEN m.value_as_number / 1000
					  ELSE m.value_as_number
				  END AS neutrophils_count
				FROM {base_query} t1
				LEFT JOIN `{dataset_project}.{dataset}.measurement` m 
				  ON t1.person_id = m.person_id
				  AND m.unit_concept_id in (
					  8848, -- 1000/uL
					  8554, -- %
					  8784, -- cells/uL
					  8961 -- 1000/mm^3
				  )
				INNER JOIN neutrophils_concepts mc 
				  ON m.measurement_concept_id = mc.concept_id
				LEFT JOIN all_wbc wbc
				  ON m.person_id = wbc.person_id 
				  AND m.measurement_datetime = wbc.measurement_datetime
			),
			all_measurements AS
			(
				SELECT DISTINCT *
				FROM all_neutrophils
				FULL OUTER JOIN all_bands USING (person_id, admit_date, discharge_date, measurement_datetime)
				FULL OUTER JOIN all_wbc USING (person_id, admit_date, discharge_date, measurement_datetime)
				WHERE 
					(
						neutrophils_count IS NOT NULL
						OR bands_count IS NOT NULL
						OR wbc IS NOT NULL
					) 
			),
			all_measurements_transformed AS
			(
				SELECT person_id, admit_date, discharge_date, measurement_datetime
					,CASE
						WHEN neutrophils_count IS NOT NULL or bands_count IS NOT NULL 
							THEN IFNULL(neutrophils_count,0) + IFNULL(bands_count,0)
						WHEN wbc IS NOT NULL AND neutrophils_count IS NULL AND bands_count IS NULL
							THEN wbc
					END AS value_as_number
				FROM all_measurements
			),
			min_measurements AS
			(
				SELECT person_id, admit_date, discharge_date
					,MIN(value_as_number) AS np_min_neutrophils
				FROM all_measurements_transformed
				WHERE measurement_datetime > admit_date 
					AND measurement_datetime <= discharge_date
				GROUP BY person_id, admit_date, discharge_date
			),
			all_abnormals_during_admission_500 as
			(
				SELECT person_id, admit_date, discharge_date
					,value_as_number 
					,measurement_datetime
					,ROW_NUMBER() OVER(
						PARTITION BY person_id, admit_date, discharge_date
						ORDER BY measurement_datetime ASC
					) rn
				FROM all_measurements_transformed
				WHERE measurement_datetime > admit_date 
					AND measurement_datetime <= discharge_date 
					AND value_as_number < 0.5
			),
			first_abnormal_during_admission_500 as
			(
				SELECT person_id, admit_date, discharge_date
					,value_as_number as np_neutrophils
					,measurement_datetime as np_neutrophils_time
				FROM all_abnormals_during_admission_500
				WHERE rn = 1
			),
			np_500 AS 
			(
				SELECT m.person_id, m.admit_date, m.discharge_date
					,np_min_neutrophils
					,np_neutrophils
					,np_neutrophils_time
					,CASE WHEN np_neutrophils IS NOT NULL THEN 1 ELSE 0 END AS np_label
				FROM min_measurements m 
				LEFT JOIN first_abnormal_during_admission_500 ab USING (person_id, admit_date, discharge_date)
			),
			all_abnormals_during_admission_1000 as
			(
				SELECT person_id, admit_date, discharge_date
					,value_as_number 
					,measurement_datetime
					,ROW_NUMBER() OVER(
						PARTITION BY person_id, admit_date, discharge_date
						ORDER BY measurement_datetime ASC
					) rn
				FROM all_measurements_transformed
				WHERE measurement_datetime > admit_date 
					AND measurement_datetime <= discharge_date 
					AND value_as_number < 1
			),
			first_abnormal_during_admission_1000 as
			(
				SELECT person_id, admit_date, discharge_date
					,value_as_number as np_neutrophils
					,measurement_datetime as np_neutrophils_time
				FROM all_abnormals_during_admission_1000
				WHERE rn = 1
			),
			np_1000 AS 
			(
				SELECT m.person_id, m.admit_date, m.discharge_date
					,np_min_neutrophils
					,np_neutrophils
					,np_neutrophils_time
					,CASE WHEN np_neutrophils IS NOT NULL THEN 1 ELSE 0 END AS np_label
				FROM min_measurements m 
				LEFT JOIN first_abnormal_during_admission_1000 ab USING (person_id, admit_date, discharge_date)
			)
			SELECT t1.*
				,np_500.np_min_neutrophils
				,np_500.np_neutrophils as np_500_neutrophils
				,np_500.np_neutrophils_time as np_500_neutrophils_time
				,CASE 
					WHEN np_500.np_label is NULL then 0
					ELSE np_500.np_label 
				END AS np_500_label
				,np_1000.np_neutrophils as np_1000_neutrophils
				,np_1000.np_neutrophils_time as np_1000_neutrophils_time
				,CASE 
					WHEN np_1000.np_label is NULL then 0
					ELSE np_1000.np_label 
				END AS np_1000_label
			FROM {base_query} t1
			LEFT JOIN np_500 USING (person_id, admit_date, discharge_date)
			LEFT JOIN np_1000 USING (person_id, admit_date, discharge_date)
		"""

class BQFilterInpatientCohort(BQCohort):
	"""
	Filters cohort
		1. Admissions where the patient is 18 or older
		2. Sample one admission per patient

	"""

	def get_defaults(self):
		config_dict = super().get_defaults()

		config_dict["cohort_name_labeled"] = "temp_cohort_labeled"
		config_dict["cohort_name_filtered"] = "temp_cohort_filtered"
		config_dict['filter_query'] = ""

		return config_dict

	def get_base_query(self, format_query=True):
		query = """
		(
			SELECT * 
			FROM {rs_dataset_project}.{rs_dataset}.{cohort_name_labeled}
			WHERE age_in_years >= 18.0
			{filter_query}
		)
		"""
		if not format_query:
			return query
		else:
			return query.format_map(self.config_dict)

	def get_transform_query(self, format_query=True):
		query = """ 
			SELECT * EXCEPT (rnd, pos), 
			FARM_FINGERPRINT(GENERATE_UUID()) as prediction_id
			FROM (
				SELECT *, ROW_NUMBER() OVER(PARTITION BY person_id ORDER BY rnd) AS pos
				FROM (
					SELECT 
						*,
						FARM_FINGERPRINT(CONCAT(CAST(person_id AS STRING), CAST(admit_date AS STRING), CAST(discharge_date AS STRING))) as rnd
					FROM {base_query}
				)
			)
			WHERE pos = 1
			ORDER BY person_id, admit_date
		"""
		if not format_query:
			return query
		else:
			return query.format(base_query=self.get_base_query())

	def get_create_query(self, format_query=True):
		query = """ 
			CREATE OR REPLACE TABLE {rs_dataset_project}.{rs_dataset}.{cohort_name_filtered} AS
			{query}
		"""
		if not format_query:
			return query
		else:
			return query.format_map(
				{**self.config_dict, **{"query": self.get_transform_query()}}
			)
