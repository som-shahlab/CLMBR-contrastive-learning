import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import os
import sys

from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split


class Featurizer(ABC):
    def __init__(self):
        pass


class BinaryHistoryFeaturizer(Featurizer):
    def __init__(self, data_dir):
        super().__init__()
        self.data_dir = data_dir

    def transform(
        self,
        patient_col="patient_id",
        time_col="index_date",
        concept_col="concept_id",
        include_tables=None,
        ignore_tables=None,
    ):
        if include_tables is None:
            tables = [x for x in os.listdir(self.data_dir)]
        else:
            tables = include_tables
        if ignore_tables is not None:
            tables = [x for x in tables if x not in ignore_tables]
        print(tables)
        data_dict = {
            table: pd.read_parquet(os.path.join(self.data_dir, table))
            for table in tables
        }
        unique_data = {
            table: data_dict[table][
                [patient_col, time_col, concept_col]
            ].drop_duplicates()
            for table in data_dict.keys()
        }
        return pd.concat(unique_data, ignore_index=True)[
            [patient_col, time_col, concept_col]
        ].drop_duplicates()
