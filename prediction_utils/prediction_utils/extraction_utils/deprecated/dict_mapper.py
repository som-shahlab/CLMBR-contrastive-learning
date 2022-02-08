import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import os
import sys
import copy
import scipy as sp

from . import featurizer, vocabulary, data_splitter


class DictMapper:
    def __init__(self, data_df, label_df, vocab_df, split_df, patient_col="patient_id"):

        self.label_df = label_df.merge(split_df)
        self.data_df = data_df.merge(split_df)
        self.vocab_df = vocab_df
        self.row_map_df = self.get_row_map(label_df, split_df, patient_col=patient_col)
        self.data_df = self.data_df.merge(self.row_map_df)
        self.label_df = self.label_df.merge(self.row_map_df)

    @staticmethod
    def get_row_map(
        label_df,
        split_df,
        patient_col="patient_id",
        date_col="index_date",
        split_col="split",
    ):
        row_map_df = label_df.merge(split_df)[
            [patient_col, date_col, split_col]
        ].drop_duplicates()
        temp = (
            row_map_df.groupby(split_col)
            .transform(lambda x: np.arange(len(x)))[[patient_col]]
            .rename(columns={patient_col: "row_id"})
        )
        row_map_df = row_map_df.merge(temp, left_index=True, right_index=True)
        return row_map_df

    @staticmethod
    def get_sparse_mat(the_df, vocab_df):
        ncol = vocab_df.shape[0]
        nrow = the_df.row_id.max() + 1
        row = the_df.row_id.values
        col = the_df.token_id.values
        values = np.ones(row.shape[0])
        the_mat = sp.sparse.csr_matrix((values, (row, col)), shape=(nrow, ncol))
        return the_mat

    def get_data_dict(self):
        splits = self.data_df["split"].unique()
        data_dict = {
            split: self.get_sparse_mat(
                self.data_df.loc[self.data_df["split"] == split], self.vocab_df
            )
            for split in splits
        }
        return data_dict

    # @staticmethod
    # def get_sparse_mat_test(the_df, vocab_df):
    #     ncol = vocab_df.shape[0]
    #     nrow = the_df.row_id.max() + 1
    #     row = the_df.row_id.values
    #     col = the_df.token_id.values
    #     values = np.ones(row.shape[0])
    #     # print(values.unique())
    #     the_mat = sp.sparse.csr_matrix((values, (row, col)), shape = (nrow, ncol))
    #     return the_mat, row, col, values

    # def get_data_dict_test(self):
    #     splits = self.data_df['split'].unique()
    #     data_dict = {}
    #     row_dict = {}
    #     col_dict = {}
    #     values_dict = {}

    #     for split in splits:
    #         data_dict[split], row_dict[split], col_dict[split], values_dict[split] = self.get_sparse_mat_test(self.data_df.loc[self.data_df['split'] == split], self.vocab_df)
    #     # data_dict = {split: self.get_sparse_mat(self.data_df.loc[self.data_df['split'] == split], self.vocab_df) for split in splits}
    #     return data_dict, row_dict, col_dict, values_dict

    def get_label_dict(
        self, label_cols=["readmission_30", "inhospital_mortality", "los_7"]
    ):
        splits = self.label_df["split"].unique()
        label_dict = {
            label_col: {
                split: np.int32(
                    self.label_df.loc[self.label_df["split"] == split]
                    .sort_values("row_id")[label_col]
                    .values
                )
                for split in splits
            }
            for label_col in label_cols
        }
        return label_dict

    def get_row_map_df(self):
        return self.row_map_df
