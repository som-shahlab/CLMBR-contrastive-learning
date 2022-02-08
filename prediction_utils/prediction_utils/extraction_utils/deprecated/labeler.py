import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import sqlalchemy as sa
import os
from sqlalchemy import create_engine
from abc import ABC, abstractmethod

from .util import *
from .database import *


class Labeler(ABC):
    """
    Defines a cohort and a labeling function
    """

    def __init__(self, label_path=None, overwrite_labels=False, db_name="stride8"):
        self.label_path = label_path
        if label_path is None:
            raise ValueError("label_path must be specified")

        if (os.path.exists(label_path)) & (not overwrite_labels):
            pass
        else:
            self.db = Database(db_name=db_name)
            self.user_string = "user_" + self.db.config_dict["user"]

    @abstractmethod
    def get_cohort(self):
        """
        Returns patient ids and index datetimes
        """


class InpatientLabelerStride8Visit(Labeler):
    """
    Labeler for inpatient admissions from Stride8. Extracts from both SHC and LPCH.
    This extraction is based on hospital accounts and gives different results than one derived from the visits
    """

    def __init__(self, label_path=None, overwrite_labels=False, db_name="stride8"):
        super().__init__(label_path, overwrite_labels, db_name)

        if (os.path.exists(label_path)) & (not overwrite_labels):
            self.cohort = pd.read_parquet(os.path.join(label_path), engine="pyarrow")
        else:
            self.query_dict = self.get_queries()
            self.cohort = self.get_cohort()
            self.cohort = self.get_labels(self.cohort)
            if overwrite_labels and os.path.exists(label_path):
                shutil.rmtree(label_path)
            pq.write_to_dataset(
                table=pa.Table.from_pandas(self.cohort),
                root_path=label_path,
                preserve_index=False,
            )

    def get_labels(self, df):
        """
        Applies the labeling functions
        """
        labelers = [self.get_readmission, self.get_mortality, self.get_los]
        for labeler in labelers:
            df = labeler(df)
        return df

    def get_cohort(self):
        df_SHC = pd.read_sql_query(
            self.query_dict["SHC"],
            self.db.engine,
            parse_dates=["inp_adm_date", "hosp_disch_time"],
        )
        df_LPCH = pd.read_sql_query(
            self.query_dict["LPCH"],
            self.db.engine,
            parse_dates=["inp_adm_date", "hosp_disch_time"],
        )
        df = pd.merge(df_SHC, df_LPCH, how="outer")
        df = df.assign(
            age_at_inp_adm_date_in_days=lambda x: np.floor(
                x.age_at_inp_adm_date_in_days
            ),
            age_at_hosp_disch_in_days=lambda x: np.floor(x.age_at_hosp_disch_in_days),
            inp_adm_date=lambda x: x.inp_adm_date.dt.floor("D"),
            hosp_disch_time=lambda x: x.hosp_disch_time.dt.floor("D"),
        )
        df = df.assign(
            los=lambda x: x.age_at_hosp_disch_in_days - x.age_at_inp_adm_date_in_days
        )
        df = df.assign(
            index_date=lambda x: x.inp_adm_date + pd.DateOffset(days=1),
            age_at_index_date=lambda x: x.age_at_inp_adm_date_in_days + 1,
        )
        df = df.loc[df.los > 0.0].drop_duplicates(["patient_id", "index_date"])
        return df

    def get_queries(self):
        query_dict = {
            "SHC": """
                                        SELECT patient_id, visit_id, hsp_account_id, 
                                            age_at_inp_adm_date_in_days, inp_adm_date, 
                                            age_at_hosp_disch_in_days, hosp_disch_time
                                        FROM SHC_visit_de
                                        WHERE age_at_inp_adm_date_in_days != 0 AND 
                                            adt_pat_class_c = 126 AND 
                                            visit_id != 0 AND 
                                            age_at_hosp_disch_in_days != 0 AND 
                                            hsp_account_id != 0 AND
                                            FLOOR(age_at_hosp_disch_in_days) > FLOOR(age_at_inp_adm_date_in_days) AND
                                            age_at_hosp_disch_in_days != 32872.5
                                    """,
            "LPCH": """
                                        SELECT patient_id, visit_id, hsp_account_id, 
                                            age_at_inp_adm_date_in_days, inp_adm_date, 
                                            age_at_hosp_disch_in_days, hosp_disch_time
                                        FROM LPCH_visit_de
                                        WHERE inp_adm_date != '0000-00-00' AND 
                                            adt_pat_class_c = 101 AND 
                                            visit_id != 0 AND 
                                            age_at_hosp_disch_in_days != 0 AND 
                                            hsp_account_id != 0 AND
                                            FLOOR(age_at_hosp_disch_in_days) > FLOOR(age_at_inp_adm_date_in_days) AND
                                            age_at_hosp_disch_in_days != 32872.5
                                    """,
        }
        return query_dict

    def get_readmission(self, df):
        """
        30 Day Readmission Labeler
        """
        ## Join patients with themselves
        df_self = pd.merge(df, df, on="patient_id", how="inner")
        # Remove self joined visits
        df_self = df_self[df_self.visit_id_x != df_self.visit_id_y]
        # Define a readmission window
        df_self = df_self.assign(
            readmission_window=lambda x: x.age_at_inp_adm_date_in_days_y
            - x.age_at_inp_adm_date_in_days_x
        )
        # Create before/after
        df_self = df_self.loc[df_self.readmission_window >= 0]
        # Take minimum readmission window so that visits are references next one observed
        temp = df_self.groupby(["patient_id", "visit_id_x"], as_index=False)[
            "readmission_window"
        ].agg(np.min)
        temp = temp.rename(columns={"visit_id_x": "visit_id"})
        df = pd.merge(df, temp, how="left")
        # Assign the label
        df = df.assign(
            readmission_30=lambda x: (x.readmission_window <= 30)
            & (x.readmission_window.notna())
        )
        return df

    def get_mortality(self, df):
        """
        Mortality Labeler
        """
        demographics = pd.read_sql_query(
            "SELECT patient_id, age_at_death_in_days FROM demographics", self.db.engine
        )
        df = pd.merge(df, demographics, on="patient_id", how="inner")
        df = df.assign(age_at_death_in_days=lambda x: np.floor(x.age_at_death_in_days))
        df = df.assign(
            inhospital_mortality=lambda x: (x.age_at_death_in_days != 0)
            & (x.age_at_death_in_days >= x.age_at_inp_adm_date_in_days)
            & (x.age_at_death_in_days <= x.age_at_hosp_disch_in_days)
        )
        return df

    def get_los(self, df):
        """
        Length of Stay Labeler
        """
        df = df.assign(los_7=lambda x: x.los >= 7.0)
        return df


class InpatientLabelerStride8Account(Labeler):
    """
    Labeler for inpatient admissions from Stride8. Extracts from both SHC and LPCH.
    This extraction is based on hospital accounts and gives different results than one derived from the visits
    """

    def __init__(self, label_path=None, overwrite_labels=False, db_name="stride8"):
        super().__init__(label_path, overwrite_labels, db_name)

        if (os.path.exists(label_path)) & (not overwrite_labels):
            self.cohort = pd.read_parquet(os.path.join(label_path), engine="pyarrow")
        else:
            self.query_dict = self.get_queries()
            self.cohort = self.get_cohort()
            self.cohort = self.get_labels(self.cohort)
            if overwrite_labels and os.path.exists(label_path):
                shutil.rmtree(label_path)
            pq.write_to_dataset(
                table=pa.Table.from_pandas(self.cohort),
                root_path=label_path,
                preserve_index=False,
            )

    def get_labels(self, df):
        """
        Applies the labeling functions
        """
        labelers = [self.get_readmission, self.get_mortality, self.get_los]
        for labeler in labelers:
            df = labeler(df)
        return df

    def get_cohort(self):
        """
        Extracts the cohort
        """
        ## SHC
        self.db.engine.execute(self.query_dict["SHC_temp"])
        SHC_dx_admit_acct_de = pd.read_sql_query(
            self.query_dict["SHC_dx_admit_acct_de_query"], self.db.engine
        )
        SHC_dx_hsp_acct_de = pd.read_sql_query(
            self.query_dict["SHC_dx_hsp_acct_de_query"], self.db.engine
        )
        SHC_admit_disch_times = pd.merge(
            SHC_dx_admit_acct_de, SHC_dx_hsp_acct_de, how="outer"
        )
        SHC_admit_disch_times = SHC_admit_disch_times.assign(
            age_at_admit_in_days=lambda x: np.floor(x.age_at_admit_in_days),
            age_at_disch_in_days=lambda x: np.floor(x.age_at_disch_in_days),
            admit_date_time=lambda x: x.admit_date_time.dt.floor("D"),
            disch_date_time=lambda x: x.disch_date_time.dt.floor("D"),
        )
        SHC_admit_disch_times = SHC_admit_disch_times.assign(
            index_date=lambda x: x.admit_date_time + pd.DateOffset(days=1),
            age_at_index_date=lambda x: x.age_at_admit_in_days + 1,
        )

        SHC_admit_disch_times = SHC_admit_disch_times.assign(
            los=lambda x: x.age_at_disch_in_days - x.age_at_admit_in_days
        )
        SHC_admit_disch_times = SHC_admit_disch_times.loc[
            SHC_admit_disch_times.los > 0.0, :
        ].drop_duplicates()
        result_SHC = SHC_admit_disch_times
        # LPCH
        self.db.engine.execute(self.query_dict["LPCH_temp"])
        LPCH_dx_admit_acct_de = pd.read_sql_query(
            self.query_dict["LPCH_dx_admit_acct_de_query"], self.db.engine
        )
        LPCH_dx_hsp_acct_de = pd.read_sql_query(
            self.query_dict["LPCH_dx_hsp_acct_de_query"], self.db.engine
        )
        LPCH_admit_disch_times = pd.merge(
            LPCH_dx_admit_acct_de, LPCH_dx_hsp_acct_de, how="outer"
        )
        LPCH_admit_disch_times = LPCH_admit_disch_times.assign(
            age_at_admit_in_days=lambda x: np.floor(x.age_at_admit_in_days),
            age_at_disch_in_days=lambda x: np.floor(x.age_at_disch_in_days),
            admit_date_time=lambda x: x.admit_date_time.dt.floor("D"),
            disch_date_time=lambda x: x.disch_date_time.dt.floor("D"),
        )

        LPCH_admit_disch_times = LPCH_admit_disch_times.assign(
            index_date=lambda x: x.admit_date_time + pd.DateOffset(days=1),
            age_at_index_date=lambda x: x.age_at_admit_in_days + 1,
        )
        LPCH_admit_disch_times = LPCH_admit_disch_times.assign(
            los=lambda x: x.age_at_disch_in_days - x.age_at_admit_in_days
        )
        LPCH_admit_disch_times = LPCH_admit_disch_times.loc[
            LPCH_admit_disch_times.los > 0.0, :
        ].drop_duplicates()
        result_LPCH = LPCH_admit_disch_times
        result = pd.merge(result_SHC, result_LPCH, how="outer").drop_duplicates()
        # Filter out 90+ year olds
        result = result.loc[
            (result.age_at_admit_in_days != 32872)
            & (result.age_at_disch_in_days != 32872)
        ]

        return result

    def get_queries(self):
        """
        Queries that define the cohort - helper for get_cohort
        """
        query_dict = {
            "SHC_temp": """
                                    CREATE TEMPORARY TABLE {}.SHC_inpatient_account
                                    (INDEX temp_index (patient_id, hsp_account_id))
                                    SELECT x.patient_id, x.hsp_account_id, prim_visit_id, age_at_admit_in_days
                                    FROM SHC_account_de x
                                    INNER JOIN SHC_visit_de AS y 
                                      ON x.patient_id = y.patient_id AND
                                         x.hsp_account_id = y.hsp_account_id
                                    WHERE acct_class_ha_c = 126 AND 
                                        prim_visit_id != 0 AND 
                                        visit_id != 0 AND
                                        age_at_inp_adm_date_in_days != 0 AND 
                                        y.hsp_account_id != 0
                                """.format(
                self.user_string
            ),
            "SHC_dx_admit_acct_de_query": """
                                                    SELECT DISTINCT x.patient_id, x.hsp_account_id,
                                                        admit_date_time, x.age_at_admit_in_days, 
                                                        disch_date_time, age_at_disch_in_days
                                                    FROM SHC_dx_admit_acct_de x
                                                    INNER JOIN {}.SHC_inpatient_account AS y
                                                        ON x.patient_id = y.patient_id AND 
                                                           x.hsp_account_id = y.hsp_account_id AND
                                                           x.age_at_admit_in_days = y.age_at_admit_in_days
                                                    WHERE age_at_disch_in_days != 0
                                                """.format(
                self.user_string
            ),
            "SHC_dx_hsp_acct_de_query": """
                                                    SELECT DISTINCT x.patient_id, x.hsp_account_id,
                                                        admit_date_time, x.age_at_admit_in_days,
                                                        disch_date_time, age_at_disch_in_days
                                                    FROM SHC_dx_hsp_acct_de x
                                                    INNER JOIN {}.SHC_inpatient_account AS y
                                                        ON x.patient_id = y.patient_id AND 
                                                           x.hsp_account_id = y.hsp_account_id AND
                                                           x.age_at_admit_in_days = y.age_at_admit_in_days
                                                    WHERE age_at_disch_in_days != 0
                                                """.format(
                self.user_string
            ),
            "LPCH_temp": """
                                    CREATE TEMPORARY TABLE {}.LPCH_inpatient_account
                                        (INDEX temp_index (patient_id, hsp_account_id))
                                        SELECT x.patient_id, x.hsp_account_id, prim_visit_id, age_at_admit_in_days
                                        FROM LPCH_account_de x
                                        INNER JOIN LPCH_visit_de AS y 
                                            ON x.patient_id = y.patient_id AND
                                               x.hsp_account_id = y.hsp_account_id
                                        WHERE   acct_class_ha_c = 101 AND 
                                                prim_visit_id != 0 AND 
                                                visit_id != 0 AND
                                                age_at_inp_adm_date_in_days != 0 AND 
                                                y.hsp_account_id != 0
                                """.format(
                self.user_string
            ),
            "LPCH_dx_admit_acct_de_query": """
                                                    SELECT DISTINCT x.patient_id, x.hsp_account_id,
                                                        admit_date_time, x.age_at_admit_in_days, 
                                                        disch_date_time, age_at_disch_in_days
                                                    FROM LPCH_dx_admit_acct_de x
                                                    INNER JOIN {}.LPCH_inpatient_account AS y
                                                        ON x.patient_id = y.patient_id AND 
                                                           x.hsp_account_id = y.hsp_account_id AND
                                                           x.age_at_admit_in_days = y.age_at_admit_in_days
                                                    WHERE age_at_disch_in_days != 0
                                                """.format(
                self.user_string
            ),
            "LPCH_dx_hsp_acct_de_query": """
                                                    SELECT DISTINCT x.patient_id, x.hsp_account_id,
                                                        admit_date_time, x.age_at_admit_in_days,
                                                        disch_date_time, age_at_disch_in_days
                                                    FROM LPCH_dx_hsp_acct_de x
                                                    INNER JOIN {}.LPCH_inpatient_account AS y
                                                        ON x.patient_id = y.patient_id AND 
                                                           x.hsp_account_id = y.hsp_account_id AND
                                                           x.age_at_admit_in_days = y.age_at_admit_in_days
                                                    WHERE age_at_disch_in_days != 0
                                                """.format(
                self.user_string
            ),
        }
        return query_dict

    def get_readmission(self, df):
        """
        30 Day Readmission Labeler
        """
        ## Join patients with themselves
        df_self = pd.merge(df, df, on="patient_id", how="inner")
        # Remove self joined visits
        df_self = df_self[df_self.hsp_account_id_x != df_self.hsp_account_id_y]
        # Define a readmission window
        df_self = df_self.assign(
            readmission_window=lambda x: x.age_at_admit_in_days_y
            - x.age_at_disch_in_days_x
        )
        # Create before/after
        df_self = df_self.loc[df_self.readmission_window >= 0]
        # Take minimum readmission window so that visits are references next one observed
        temp = df_self.groupby(["patient_id", "hsp_account_id_x"], as_index=False)[
            "readmission_window"
        ].agg(np.min)
        temp = temp.rename(columns={"hsp_account_id_x": "hsp_account_id"})
        df = pd.merge(df, temp, how="left")
        # Assign the label
        df = df.assign(
            readmission_30=lambda x: (x.readmission_window <= 30)
            & (x.readmission_window.notna())
        )
        return df

    def get_mortality(self, df):
        """
        Mortality Labeler
        """
        demographics = pd.read_sql_query(
            "SELECT patient_id, age_at_death_in_days FROM demographics", self.db.engine
        )
        df = pd.merge(df, demographics, on="patient_id", how="inner")
        df = df.assign(age_at_death_in_days=lambda x: np.floor(x.age_at_death_in_days))
        df = df.assign(
            inhospital_mortality=lambda x: (x.age_at_death_in_days != 0)
            & (x.age_at_death_in_days >= x.age_at_admit_in_days)
            & (x.age_at_death_in_days <= x.age_at_disch_in_days)
        )
        return df

    def get_los(self, df):
        """
        Length of Stay Labeler
        """
        df = df.assign(los_7=lambda x: x.los >= 7.0)
        return df
