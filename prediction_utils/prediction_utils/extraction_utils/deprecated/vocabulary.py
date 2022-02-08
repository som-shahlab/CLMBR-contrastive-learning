import numpy as np
import pandas as pd


class Vocabulary:
    """
    TO-DO
        Provide existing dictionary
        Merge two vocabularies
        Inverse Transform
    """

    def __init__(self, vocab_df=None, word_col="concept_id", token_col="token_id"):
        self.vocab_df = None
        self.word_col = word_col
        self.token_col = token_col

    def train(self, df):
        vocab_df = df.loc[:, [self.word_col]].drop_duplicates()
        self.vocab_size = vocab_df.shape[0]
        vocab_df[self.token_col] = np.arange(self.vocab_size)
        self.vocab_df = vocab_df

    def transform(self, df, word_col="concept_id", how="inner"):
        if self.vocab_df is None:
            raise ValueError("Please provide vocab_df or train before transforming")
        if how not in ["left", "inner"]:
            raise ValueError("Only left and inner joins supported")
        return pd.merge(
            df, self.vocab_df, left_on=word_col, right_on=self.word_col, how=how
        )
