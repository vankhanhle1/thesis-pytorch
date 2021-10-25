import pandas as pd

from helpers import remove_stopwords_and_special_characters


class Data:
    def __init__(self, max_len=0):
        self.s0s, self.s1s, self.scores = [], [], []
        self.processed = []
        self.embedded = []
        self.max_len = max_len

    def open_and_preprocess(self):
        pass


class STS(Data):
    def open_and_preprocess(self, path, mode):
        print("Removing stopwords")
        df = pd.read_csv(path + "-" + mode + "-df.csv")
        for i in range(len(df)):
            s0 = remove_stopwords_and_special_characters(df.iloc[i, 0])
            s1 = remove_stopwords_and_special_characters(df.iloc[i, 1])
            score = df.iloc[i, 2]
            row = [s0, s1, score]
            self.processed.append(row)
            local_max_len = max(len(s0), len(s1))
            if local_max_len > self.max_len:
                self.max_len = local_max_len
