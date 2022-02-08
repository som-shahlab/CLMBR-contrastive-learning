import numpy as np
import pandas as pd
import copy

from sklearn.model_selection import train_test_split
from abc import ABC, abstractmethod


class DataSplitter(ABC):
    """
    Splits Data
    """

    def __init__(self):
        pass

    @abstractmethod
    def split(self):
        """
        Splits the data
        """


class PatientSplitter(DataSplitter):
    def __init__(self, seeds=[527, 196]):
        super().__init__()
        self.seeds = seeds

    def split(
        self,
        df,
        split_col="patient_id",
        split_dict={"train": 0.8, "val": 0.1, "test": 0.1},
    ):

        df_unique = df[[split_col]].drop_duplicates()
        train_size = split_dict["train"]
        val_test_size = split_dict["val"] + split_dict["test"]
        val_size = split_dict["val"] / val_test_size
        test_size = split_dict["test"] / val_test_size
        if (train_size + val_test_size) != 1.0:
            raise ValueError("Size of sets must add to 1.0")
        tuple_train = train_test_split(
            df_unique,
            train_size=train_size,
            test_size=val_test_size,
            random_state=self.seeds[0],
        )
        tuple_val_test = train_test_split(
            tuple_train[1],
            train_size=val_size,
            test_size=test_size,
            random_state=self.seeds[1],
        )

        df_train = copy.deepcopy(tuple_train[0])
        df_val = copy.deepcopy(tuple_val_test[0])
        df_test = copy.deepcopy(tuple_val_test[1])

        df_train.loc[:, "split"] = "train"
        df_val.loc[:, "split"] = "val"
        df_test.loc[:, "split"] = "test"
        patient_df = pd.concat((df_train, df_val, df_test), ignore_index=True)
        df = pd.merge(df, patient_df, on=split_col, how="left")
        return df, patient_df


class TimelinePatientSplitter(DataSplitter):
    def __init__(self, seed=937):
        super().__init__()
        self.seed = seed

    def split(
        self, df, split_col="patient_id", time_col="index_date", split_date="2017-01-01"
    ):
        ## Find patients who have visits after the split date
        patient_df = df[[split_col]].drop_duplicates()
        df_post = df.loc[df.index_date >= split_date]
        df_val_test = df_post[[split_col]].drop_duplicates()
        tuples_val_test = train_test_split(
            df_val_test, train_size=0.5, test_size=0.5, random_state=self.seed
        )
        df_val = copy.deepcopy(tuples_val_test[0])
        df_test = copy.deepcopy(tuples_val_test[1])
        df_val.loc[:, "split"] = "val"
        df_test.loc[:, "split"] = "test"
        df_val_test = pd.concat((df_val, df_test), ignore_index=True)
        patient_df = patient_df.merge(df_val_test, on=split_col, how="left")
        patient_df.loc[pd.isna(patient_df.split), "split"] = "train"
        df = df.merge(patient_df, on=split_col, how="left")
        return df, patient_df
