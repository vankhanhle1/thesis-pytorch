import pandas as pd
import numpy as np
from rouge_score import rouge_scorer

from data_helpers import remove_special_characters, StopWordsRemoval, Word2Vec, extract_mt_features


class Data:
    def __init__(self, max_len=0):
        self.s0s, self.s1s, self.scores = [], [], []
        self.original = []
        self.processed = []
        self.embedded = []
        self.max_len = max_len
        self.terp_path = ''

    def open_and_preprocess(self, path, mode):
        print("Removing special chars & stopwords")
        df = pd.read_csv(path + mode + "-df.csv")
        self.terp_path = path + "terp\\" + mode + "\\msrp.seg.csv"
        for i in range(len(df)):
            s0_original, s1_original = df.iloc[i, 0], df.iloc[i, 1]
            self.original.append([s0_original, s1_original])
            s0 = remove_special_characters(s0_original)
            s1 = remove_special_characters(s1_original)
            score = df.iloc[i, 2]
            self.processed.append([s0, s1, score])
            local_max_len = max(len(s0), len(s1))

            # s0 = stop_words_removal.remove_stopwords(s0_original)
            # s1 = stop_words_removal.remove_stopwords(s1_original)
            # score = df.iloc[i, 2]
            # self.processed.append([s0, s1, score])
            # local_max_len = max(len(s0), len(s1))

            if local_max_len > self.max_len:
                self.max_len = local_max_len

    def get_embeddings(self, embedding_type):
        # embedding_types = {"word2vec": getWord2Vec}
        embed = Word2Vec()
        rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2'])
        terp = pd.read_csv(self.terp_path)
        print("Getting embeddings")
        for i in range(len(self.processed)):
            if (i + 1) % 100 == 0:
                print("Getting embeddings for pair " + str(i + 1) + " out of " + str(len(self.processed) + 1))
            s0, s1, score = self.processed[i]
            s0_embedded = np.pad(np.column_stack([embed.get(w) for w in s0]), [[0, 0], [0, self.max_len - len(s0)]],
                                 "constant")
            s1_embedded = np.pad(np.column_stack([embed.get(w) for w in s1]), [[0, 0], [0, self.max_len - len(s1)]],
                                 "constant")
            s0_original, s1_original = self.original[i]
            # BLEU14_NIST15_METEOR_ROUGE12
            additional_features_01 = extract_mt_features(s0_original, s1_original, rouge)
            additional_features_10 = extract_mt_features(s1_original, s0_original, rouge)
            # TERp
            additional_features_01.append(terp.iloc[i * 2, 1])
            additional_features_10.append(terp.iloc[i * 2 + 1, 1])
            # sentence lengths
            additional_features_01.append(len(s0))
            additional_features_01.append(len(s1))
            additional_features_01.append(score)
            additional_features_10.append(len(s1))
            additional_features_10.append(len(s0))
            additional_features_10.append(score)

            self.embedded.append([s0_embedded, s1_embedded, additional_features_01])
            self.embedded.append([s1_embedded, s0_embedded, additional_features_10])
