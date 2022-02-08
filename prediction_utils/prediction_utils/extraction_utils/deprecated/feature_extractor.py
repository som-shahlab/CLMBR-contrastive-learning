import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import sqlalchemy as sa
import os
import shutil

from sqlalchemy import create_engine
from abc import ABC, abstractmethod

from .util import *
from .database import *


class Extractor(ABC):
    """
    Extracts data
    """

    def __init__(
        self,
        df=None,
        db_name="cdmv5_stride8_strict",
        cohort_name="my_cohort",
        patient_var="patient_id",
        index_date_var="index_date",
    ):

        self.cohort_name = cohort_name

    @abstractmethod
    def extract(self, write_path, chunksize):
        """
        Extract method
        """


class SQLExtractor(Extractor):
    """
    Abstract class that defines an extractor that pulls data from an SQL database.
    """

    def __init__(
        self,
        df=None,
        db_name="cdmv5_stride8_strict",
        cohort_name="my_cohort",
        user_string=None,
    ):
        super().__init__(df, db_name, cohort_name)

        self.db = Database(db_name)
        if user_string is None:
            self.user_string = "user_" + self.db.config_dict["user"]
        else:
            self.user_string = user_string
        if df is not None:
            tuples = self.db.df_to_tuples(df)
            # self.db.insert_into_db(tuples, cohort_name = cohort_name)
        self.query_dict = self.get_queries()

    @abstractmethod
    def get_queries(self):
        """
        Dictionaries of queries
        """

    def extract(self, write_path, tables=None, chunksize=int(1e7)):

        for table, query in self.query_dict.items():
            print(table)
            if tables is None:
                self.db.stream_query(
                    table=table,
                    engine=self.db.engine,
                    query_dict=self.query_dict,
                    write_path=write_path,
                    chunksize=chunksize,
                )
            elif table in tables:
                self.db.stream_query(
                    table=table,
                    engine=self.db.engine,
                    query_dict=self.query_dict,
                    write_path=write_path,
                    chunksize=chunksize,
                )
            else:
                pass


class Stride8Extractor(SQLExtractor):
    """
    Extractor for native Stride 8
    """

    def __init__(self, df=None, db_name="stride8", cohort_name="my_cohort"):
        super().__init__(df, db_name, cohort_name)

    def get_queries(self):
        query_dict = {
            "dx_master": """
                                SELECT t1.patient_id, 
                                    index_date, 
                                    CONCAT(code_source, '_', code) AS concept_id, 
                                    contact_date_time AS concept_date, 
                                    'dx_master' as domain
                                FROM dx_master t1
                                INNER JOIN {}.{} as t2 ON
                                    t1.patient_id = t2.patient_id
                                WHERE contact_date_time < t2.index_date AND
                                    code_source = 'DX_ID' AND 
                                    contact_date_time != '0000-00-00 00:00:00'
                                    -- limit 10000
                            """.format(
                self.user_string, self.cohort_name
            ),
            "px_master": """
                                SELECT t1.patient_id, 
                                    index_date, 
                                    CONCAT(code_source, '_', code) AS concept_id, 
                                    contact_date AS concept_date, 
                                    'px_master' as domain
                                  FROM px_master t1
                                  INNER JOIN {}.{} as t2 ON
                                      t1.patient_id = t2.patient_id
                                  WHERE contact_date < t2.index_date AND
                                  contact_date != '0000-00-00'
                                  -- limit 10000
                            """.format(
                self.user_string, self.cohort_name
            ),
            "LPCH_med_de": """
                                SELECT t1.patient_id,
                                    index_date,
                                    CONCAT('medication_id_', medication_id) AS concept_id,
                                    order_time AS concept_date,
                                    'med_de' AS domain
                                FROM LPCH_med_de t1
                                INNER JOIN {}.{} as t2 ON
                                    t1.patient_id = t2.patient_id
                                WHERE order_time < t2.index_date 
                                    AND order_time != '0000-00-00 00:00:00'
                                -- limit 10000
                            """.format(
                self.user_string, self.cohort_name
            ),
            "SHC_med_de": """
                                SELECT t1.patient_id,
                                    index_date,
                                    CONCAT('medication_id_', medication_id) AS concept_id,
                                    order_time AS concept_date,
                                    'med_de' AS domain
                                FROM SHC_med_de t1
                                INNER JOIN {}.{} as t2 ON
                                    t1.patient_id = t2.patient_id
                                WHERE order_time < t2.index_date
                                    AND order_time != '0000-00-00 00:00:00'
                                -- limit 10000
                            """.format(
                self.user_string, self.cohort_name
            ),
            "SHC_lab_epic_de": """
                                    SELECT t1.patient_id,
                                        index_date,
                                        CONCAT('LOINC_', LOINC) AS concept_id,
                                        order_time as concept_date,
                                        'lab' AS domain
                                    FROM SHC_lab_epic_de t1
                                    INNER JOIN {}.{} as t2 ON
                                        t1.patient_id = t2.patient_id
                                    WHERE LOINC is not NULL AND
                                        order_time < t2.index_date
                                        AND order_time != '0000-00-00 00:00:00'
                                    -- limit 10000
                                """.format(
                self.user_string, self.cohort_name
            ),
            "LPCH_lab_epic_de": """
                                        SELECT t1.patient_id,
                                            index_date,
                                            CONCAT('LOINC_', LOINC) AS concept_id,
                                            order_time as concept_date,
                                            'lab' AS domain
                                        FROM LPCH_lab_epic_de t1
                                        INNER JOIN {}.{} as t2 ON
                                            t1.patient_id = t2.patient_id
                                        WHERE LOINC is not NULL AND
                                            order_time < t2.index_date
                                            AND order_time != '0000-00-00 00:00:00'
                                        -- limit 10000
                                    """.format(
                self.user_string, self.cohort_name
            ),
            "LPCH_lab_de": """
                               SELECT t1.patient_id,
                                   index_date,
                                   CONCAT('LOINC_', LOINC) AS concept_id,
                                   lab_time as concept_date,
                                   'lab' AS domain
                                FROM LPCH_lab_de t1
                                INNER JOIN {}.{} as t2 ON
                                      t1.patient_id = t2.patient_id
                                WHERE LOINC is not NULL AND
                                    lab_time < t2.index_date
                                    AND lab_time != '0000-00-00'
                                -- limit 10000
                            """.format(
                self.user_string, self.cohort_name
            ),
            "notes_master": """
                                SELECT t1.patient_id,
                                    index_date,
                                    CONCAT('note_type_', note_type) AS concept_id,
                                    note_date as concept_date,
                                    'note_type' AS domain
                                FROM notes_master t1
                                INNER JOIN {}.{} as t2 ON
                                    t1.patient_id = t2.patient_id
                                WHERE note_date < t2.index_date
                                    AND note_date != '0000-00-00 00:00:00'
                                -- limit 10000
                                """.format(
                self.user_string, self.cohort_name
            ),
            "terms": """
                            SELECT t1.patient_id,
                                index_date,
                                CONCAT('CUI_', CUI, '_neg_', negated, '_fam_', familyHistory) AS concept_id,
                                note_date as concept_date,
                                'note_CUIs' AS domain
                            FROM notes_master_uniques t1
                            INNER JOIN {}.{} as t2 ON
                                t1.patient_id = t2.patient_id
                            INNER JOIN term_mentions as t3 ON
                                t1.unique_note_id = t3.nid
                            INNER JOIN terminology5.str2cui as t4 ON
                                t3.tid = t4.tid
                            WHERE note_date < t2.index_date
                                AND note_date != '0000-00-00 00:00:00'
                            -- limit 10000
                        """.format(
                self.user_string, self.cohort_name
            ),
            "SHC_visit_de_enc": """
                                SELECT t1.patient_id,
                                    index_date,
                                    CONCAT('SHC_enc_type_', enc_type_c) AS concept_id,
                                    contact_date as concept_date,
                                    'enc_type' AS domain
                                FROM SHC_visit_de t1
                                INNER JOIN {}.{} as t2 ON
                                    t1.patient_id = t2.patient_id
                                WHERE contact_date < t2.index_date AND
                                    visit_id != 0
                                    AND contact_date != '0000-00-00'
                                -- limit 10000
                                """.format(
                self.user_string, self.cohort_name
            ),
            "LPCH_visit_de_enc": """
                                SELECT t1.patient_id,
                                    index_date,
                                    CONCAT('LPCH_enc_type_', enc_type_c) AS concept_id,
                                    contact_date as concept_date,
                                    'enc_type' AS domain
                                FROM LPCH_visit_de t1
                                INNER JOIN {}.{} as t2 ON
                                    t1.patient_id = t2.patient_id
                                WHERE contact_date < t2.index_date AND
                                    visit_id != 0
                                    AND contact_date != '0000-00-00'
                                -- limit 10000
                                """.format(
                self.user_string, self.cohort_name
            ),
            "SHC_visit_de_dep": """
                                SELECT t1.patient_id,
                                    index_date,
                                    CONCAT('SHC_department_id', department_id) AS concept_id,
                                    contact_date as concept_date,
                                    'department_id' AS domain
                                FROM SHC_visit_de t1
                                INNER JOIN {}.{} as t2 ON
                                    t1.patient_id = t2.patient_id
                                WHERE contact_date < t2.index_date AND
                                    visit_id != 0
                                    AND contact_date != '0000-00-00'
                                -- limit 10000
                                """.format(
                self.user_string, self.cohort_name
            ),
            "LPCH_visit_de_dep": """
                                SELECT t1.patient_id,
                                    index_date,
                                    CONCAT('LPCH_department_id', department_id) AS concept_id,
                                    contact_date as concept_date,
                                    'department_id' AS domain
                                FROM LPCH_visit_de t1
                                INNER JOIN {}.{} as t2 ON
                                    t1.patient_id = t2.patient_id
                                WHERE contact_date < t2.index_date AND
                                    visit_id != 0
                                    AND contact_date != '0000-00-00'
                                -- limit 10000
                                """.format(
                self.user_string, self.cohort_name
            ),
            ## Demographic variables - birth_date is used as concept_date
            "gender": """
                        SELECT t1.patient_id,
                            index_date,
                            CONCAT('gender_', gender) AS concept_id,
                            birth_date as concept_date,
                            'gender' AS domain
                        FROM demographics t1
                        INNER JOIN {}.{} as t2 ON
                            t1.patient_id = t2.patient_id
                        -- limit 10000
                        """.format(
                self.user_string, self.cohort_name
            ),
            "race": """
                        SELECT t1.patient_id,
                            index_date,
                            CONCAT('race_', race) AS concept_id,
                            birth_date as concept_date,
                            'race' AS domain
                        FROM demographics t1
                        INNER JOIN {}.{} as t2 ON
                            t1.patient_id = t2.patient_id
                        -- limit 10000
                        """.format(
                self.user_string, self.cohort_name
            ),
            "ethnicity": """
                        SELECT t1.patient_id,
                            index_date,
                            birth_date as concept_date,
                            CONCAT('ethnicity_', ethnicity) AS concept_id,
                            'ethnicity' AS domain
                        FROM demographics t1
                        INNER JOIN {}.{} as t2 ON
                            t1.patient_id = t2.patient_id
                        -- limit 10000
                        """.format(
                self.user_string, self.cohort_name
            ),
            ## Binning age by some logical categories. Can be improved
            "age": """
                        SELECT temp.*,
                            CASE
                                WHEN ((age >= 0.0) AND (age < 18.0)) THEN 'age_0'
                                WHEN ((age >= 18.0) AND (age < 40.0)) THEN 'age_1'
                                WHEN ((age >= 40.0) AND (age < 55.0)) THEN 'age_2'
                                WHEN ((age >= 55.0) AND (age < 65.0)) THEN 'age_3'
                                WHEN ((age >= 65.0) AND (age < 75.0)) THEN 'age_4'
                                WHEN (age >= 75.0) THEN 'age_5'
                            ELSE NULL
                            END as concept_id
                        FROM (
                            SELECT t1.patient_id,
                                index_date,
                                DATE_SUB(index_date, INTERVAL 1 DAY) as concept_date,
                                DATEDIFF(index_date, birth_date) / 365.25 as age,
                                'age' as domain
                            FROM demographics t1
                            INNER JOIN {}.{} as t2 ON
                                t1.patient_id = t2.patient_id
                        ) temp
                        -- limit 10000
                        """.format(
                self.user_string, self.cohort_name
            ),
        }
        return query_dict


class Stride8ExtractorAge(SQLExtractor):
    """
    Extractor for native Stride 8. Use index age for comparison rather than index date
    """

    def __init__(self, df=None, db_name="stride8", cohort_name="my_cohort"):
        super().__init__(df, db_name, cohort_name)

    def get_queries(self):
        query_dict = {
            "dx_master": """
                                SELECT t1.patient_id, 
                                    index_date, 
                                    CONCAT(code_source, '_', code) AS concept_id, 
                                    contact_date_time AS concept_date, 
                                    'dx_master' as domain
                                FROM dx_master t1
                                INNER JOIN {}.{} as t2 ON
                                    t1.patient_id = t2.patient_id
                                WHERE age_at_contact_in_days < t2.age_at_index_date AND
                                    code_source = 'DX_ID' AND 
                                    contact_date_time != '0000-00-00 00:00:00'
                                    -- limit 10000
                            """.format(
                self.user_string, self.cohort_name
            ),
            "px_master": """
                                SELECT t1.patient_id, 
                                    index_date, 
                                    CONCAT(code_source, '_', code) AS concept_id, 
                                    contact_date AS concept_date, 
                                    'px_master' as domain
                                  FROM px_master t1
                                  INNER JOIN {}.{} as t2 ON
                                      t1.patient_id = t2.patient_id
                                  WHERE age_at_service_in_days < t2.age_at_index_date AND
                                  contact_date != '0000-00-00'
                                  -- limit 10000
                            """.format(
                self.user_string, self.cohort_name
            ),
            "LPCH_med_de": """
                                SELECT t1.patient_id,
                                    index_date,
                                    CONCAT('medication_id_', medication_id) AS concept_id,
                                    order_time AS concept_date,
                                    'med_de' AS domain
                                FROM LPCH_med_de t1
                                INNER JOIN {}.{} as t2 ON
                                    t1.patient_id = t2.patient_id
                                WHERE age_at_order_in_days < t2.age_at_index_date 
                                    AND order_time != '0000-00-00 00:00:00'
                                -- limit 10000
                            """.format(
                self.user_string, self.cohort_name
            ),
            "SHC_med_de": """
                                SELECT t1.patient_id,
                                    index_date,
                                    CONCAT('medication_id_', medication_id) AS concept_id,
                                    order_time AS concept_date,
                                    'med_de' AS domain
                                FROM SHC_med_de t1
                                INNER JOIN {}.{} as t2 ON
                                    t1.patient_id = t2.patient_id
                                WHERE age_at_order_in_days < t2.age_at_index_date
                                    AND order_time != '0000-00-00 00:00:00'
                                -- limit 10000
                            """.format(
                self.user_string, self.cohort_name
            ),
            "SHC_lab_epic_de": """
                                    SELECT t1.patient_id,
                                        index_date,
                                        CONCAT('LOINC_', LOINC) AS concept_id,
                                        order_time as concept_date,
                                        'lab' AS domain
                                    FROM SHC_lab_epic_de t1
                                    INNER JOIN {}.{} as t2 ON
                                        t1.patient_id = t2.patient_id
                                    WHERE LOINC is not NULL AND
                                        age_at_order_in_days < t2.age_at_index_date
                                        AND order_time != '0000-00-00 00:00:00'
                                    -- limit 10000
                                """.format(
                self.user_string, self.cohort_name
            ),
            "LPCH_lab_epic_de": """
                                        SELECT t1.patient_id,
                                            index_date,
                                            CONCAT('LOINC_', LOINC) AS concept_id,
                                            order_time as concept_date,
                                            'lab' AS domain
                                        FROM LPCH_lab_epic_de t1
                                        INNER JOIN {}.{} as t2 ON
                                            t1.patient_id = t2.patient_id
                                        WHERE LOINC is not NULL AND
                                            age_at_order_in_days < t2.age_at_index_date
                                            AND order_time != '0000-00-00 00:00:00'
                                        -- limit 10000
                                    """.format(
                self.user_string, self.cohort_name
            ),
            "LPCH_lab_de": """
                               SELECT t1.patient_id,
                                   index_date,
                                   CONCAT('LOINC_', LOINC) AS concept_id,
                                   lab_time as concept_date,
                                   'lab' AS domain
                                FROM LPCH_lab_de t1
                                INNER JOIN {}.{} as t2 ON
                                      t1.patient_id = t2.patient_id
                                WHERE LOINC is not NULL AND
                                    age_at_lab_in_days < t2.age_at_index_date
                                    AND lab_time != '0000-00-00'
                                -- limit 10000
                            """.format(
                self.user_string, self.cohort_name
            ),
            "notes_master": """
                                SELECT t1.patient_id,
                                    index_date,
                                    CONCAT('note_type_', note_type) AS concept_id,
                                    note_date as concept_date,
                                    'note_type' AS domain
                                FROM notes_master t1
                                INNER JOIN {}.{} as t2 ON
                                    t1.patient_id = t2.patient_id
                                WHERE age_at_note_in_days < t2.age_at_index_date
                                    AND note_date != '0000-00-00 00:00:00'
                                -- limit 10000
                                """.format(
                self.user_string, self.cohort_name
            ),
            # 'terms' :   """
            #                 SELECT t1.patient_id,
            #                     index_date,
            #                     CONCAT('CUI_', CUI, '_neg_', negated, '_fam_', familyHistory) AS concept_id,
            #                     note_date as concept_date,
            #                     'note_CUIs' AS domain
            #                 FROM notes_master_uniques t1
            #                 INNER JOIN {}.{} as t2 ON
            #                     t1.patient_id = t2.patient_id
            #                 INNER JOIN term_mentions as t3 ON
            #                     t1.unique_note_id = t3.nid
            #                 INNER JOIN terminology5.str2cui as t4 ON
            #                     t3.tid = t4.tid
            #                 WHERE age_at_note_in_days < t2.age_at_index_date
            #                     AND note_date != '0000-00-00 00:00:00'
            #                 -- limit 10000
            #             """.format(self.user_string, self.cohort_name),
            "SHC_visit_de_enc": """
                                SELECT t1.patient_id,
                                    index_date,
                                    CONCAT('SHC_enc_type_', enc_type_c) AS concept_id,
                                    contact_date as concept_date,
                                    'enc_type' AS domain
                                FROM SHC_visit_de t1
                                INNER JOIN {}.{} as t2 ON
                                    t1.patient_id = t2.patient_id
                                WHERE age_at_contact_in_days < t2.age_at_index_date AND
                                    visit_id != 0
                                    AND contact_date != '0000-00-00'
                                -- limit 10000
                                """.format(
                self.user_string, self.cohort_name
            ),
            "LPCH_visit_de_enc": """
                                SELECT t1.patient_id,
                                    index_date,
                                    CONCAT('LPCH_enc_type_', enc_type_c) AS concept_id,
                                    contact_date as concept_date,
                                    'enc_type' AS domain
                                FROM LPCH_visit_de t1
                                INNER JOIN {}.{} as t2 ON
                                    t1.patient_id = t2.patient_id
                                WHERE age_at_contact_in_days < t2.age_at_index_date AND
                                    visit_id != 0
                                    AND contact_date != '0000-00-00'
                                -- limit 10000
                                """.format(
                self.user_string, self.cohort_name
            ),
            "SHC_visit_de_dep": """
                                SELECT t1.patient_id,
                                    index_date,
                                    CONCAT('SHC_department_id', department_id) AS concept_id,
                                    contact_date as concept_date,
                                    'department_id' AS domain
                                FROM SHC_visit_de t1
                                INNER JOIN {}.{} as t2 ON
                                    t1.patient_id = t2.patient_id
                                WHERE age_at_contact_in_days < t2.age_at_index_date AND
                                    visit_id != 0
                                    AND contact_date != '0000-00-00'
                                -- limit 10000
                                """.format(
                self.user_string, self.cohort_name
            ),
            "LPCH_visit_de_dep": """
                                SELECT t1.patient_id,
                                    index_date,
                                    CONCAT('LPCH_department_id', department_id) AS concept_id,
                                    contact_date as concept_date,
                                    'department_id' AS domain
                                FROM LPCH_visit_de t1
                                INNER JOIN {}.{} as t2 ON
                                    t1.patient_id = t2.patient_id
                                WHERE age_at_contact_in_days < t2.age_at_index_date AND
                                    visit_id != 0
                                    AND contact_date != '0000-00-00'
                                -- limit 10000
                                """.format(
                self.user_string, self.cohort_name
            ),
            ## Demographic variables - birth_date is used as concept_date
            "gender": """
                        SELECT t1.patient_id,
                            index_date,
                            CONCAT('gender_', gender) AS concept_id,
                            birth_date as concept_date,
                            'gender' AS domain
                        FROM demographics t1
                        INNER JOIN {}.{} as t2 ON
                            t1.patient_id = t2.patient_id
                        -- limit 10000
                        """.format(
                self.user_string, self.cohort_name
            ),
            "race": """
                        SELECT t1.patient_id,
                            index_date,
                            CONCAT('race_', race) AS concept_id,
                            birth_date as concept_date,
                            'race' AS domain
                        FROM demographics t1
                        INNER JOIN {}.{} as t2 ON
                            t1.patient_id = t2.patient_id
                        -- limit 10000
                        """.format(
                self.user_string, self.cohort_name
            ),
            "ethnicity": """
                        SELECT t1.patient_id,
                            index_date,
                            birth_date as concept_date,
                            CONCAT('ethnicity_', ethnicity) AS concept_id,
                            'ethnicity' AS domain
                        FROM demographics t1
                        INNER JOIN {}.{} as t2 ON
                            t1.patient_id = t2.patient_id
                        -- limit 10000
                        """.format(
                self.user_string, self.cohort_name
            ),
            ## Binning age by some logical categories. Can be improved
            "age": """
                        SELECT temp.*,
                            CASE
                                WHEN ((age >= 0.0) AND (age < 18.0)) THEN 'age_0'
                                WHEN ((age >= 18.0) AND (age < 40.0)) THEN 'age_1'
                                WHEN ((age >= 40.0) AND (age < 55.0)) THEN 'age_2'
                                WHEN ((age >= 55.0) AND (age < 65.0)) THEN 'age_3'
                                WHEN ((age >= 65.0) AND (age < 75.0)) THEN 'age_4'
                                WHEN (age >= 75.0) THEN 'age_5'
                            ELSE NULL
                            END as concept_id
                        FROM (
                            SELECT t1.patient_id,
                                index_date,
                                DATE_SUB(index_date, INTERVAL 1 DAY) as concept_date,
                                DATEDIFF(index_date, birth_date) / 365.25 as age,
                                'age' as domain
                            FROM demographics t1
                            INNER JOIN {}.{} as t2 ON
                                t1.patient_id = t2.patient_id
                        ) temp
                        -- limit 10000
                        """.format(
                self.user_string, self.cohort_name
            ),
        }
        return query_dict


class OmopDBExtractor(SQLExtractor):
    """
    Extractor for OMOP data
    """

    def __init__(
        self, df=None, db_name="cdmv5_stride8_strict", cohort_name="my_cohort"
    ):
        super().__init__(df, db_name, cohort_name)

    def get_queries(self):
        query_dict = {
            # 'visit_occurrence' : """
            #                         SELECT t1.person_id,
            #                             index_date,
            #                             visit_concept_id AS concept_id,
            #                             visit_start_datetime AS concept_date,
            #                             'visit' AS domain
            #                         FROM visit_occurrence t1
            #                         INNER JOIN {}.{} AS t2 ON
            #                             t1.person_id = t2.patient_id
            #                         WHERE visit_start_datetime < t2.index_date
            #                         -- limit 10000
            #                      """.format(self.user_string, self.cohort_name),
            "condition_occurrence": """
                                        SELECT t1.person_id, 
                                            index_date, 
                                            condition_concept_id AS concept_id, 
                                            condition_start_datetime AS concept_date,
                                            'condition' AS domain
                                        FROM condition_occurrence t1
                                        INNER JOIN {}.{} AS t2 ON
                                            t1.person_id = t2.patient_id
                                        WHERE condition_start_datetime < t2.index_date
                                        -- limit 10000
                                     """.format(
                self.user_string, self.cohort_name
            ),
            "observation": """
                                        SELECT t1.person_id, 
                                            index_date, 
                                            observation_concept_id AS concept_id, 
                                            observation_datetime AS concept_date,
                                            'observation' AS domain
                                        FROM observation t1
                                        INNER JOIN {}.{} AS t2 ON
                                            t1.person_id = t2.patient_id
                                        WHERE observation_datetime < t2.index_date
                                        -- limit 10000
                                     """.format(
                self.user_string, self.cohort_name
            ),
            "drug_exposure": """
                                        SELECT t1.person_id, 
                                            index_date, 
                                            drug_concept_id AS concept_id, 
                                            drug_exposure_start_datetime AS concept_date,
                                            'drug' AS domain
                                        FROM drug_exposure t1
                                        INNER JOIN {}.{} AS t2 ON
                                            t1.person_id = t2.patient_id
                                        WHERE drug_exposure_start_datetime < t2.index_date
                                        -- limit 10000
                                     """.format(
                self.user_string, self.cohort_name
            ),
            "measurement": """
                                        SELECT t1.person_id, 
                                            index_date, 
                                            measurement_concept_id AS concept_id, 
                                            measurement_datetime AS concept_date,
                                            'measurement' AS domain
                                        FROM measurement_v2 t1
                                        INNER JOIN {}.{} AS t2 ON
                                            t1.person_id = t2.patient_id
                                        WHERE measurement_datetime < t2.index_date
                                        -- limit 10000
                                     """.format(
                self.user_string, self.cohort_name
            ),
            "procedure_occurrence": """
                                    SELECT t1.person_id, 
                                        index_date, 
                                        procedure_concept_id AS concept_id, 
                                        procedure_datetime AS concept_date,
                                        'procedure' AS domain
                                    FROM procedure_occurrence t1
                                    INNER JOIN {}.{} AS t2 ON
                                        t1.person_id = t2.patient_id
                                    WHERE procedure_datetime < t2.index_date
                                    -- limit 10000
                                 """.format(
                self.user_string, self.cohort_name
            ),
        }
        return query_dict
