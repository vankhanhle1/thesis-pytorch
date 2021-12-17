import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer

from data_helpers import remove_special_characters, Word2Vec, extract_mt_features, remove_all_non_alphanumeric


class Data:
    def __init__(self, max_len=0):
        self.s0s, self.s1s, self.scores = [], [], []
        self.original = []
        self.processed = []
        self.embedded = []
        self.max_len = max_len
        self.terp_path = ''
        self.embeddings_only = []
        self.embeddings_and_additional_features = []


class MSRP(Data):
    def open_and_preprocess(self, path, mode):
        print("Removing special chars & stopwords")
        df = pd.read_csv(path + mode + "-df.csv")
        self.terp_path = path + "sgm\\" + mode + "\\output\\terp\\" + mode + ".seg.csv"
        self.badger_path = path + "sgm\\" + mode + "\\output\\SmithWatermanGotohWindowedAffine\\Badger-seg.csv"
        for i in range(len(df)):
            # s0_original, s1_original = df.iloc[i, 0], df.iloc[i, 1]
            s0, s1 = remove_special_characters(df.iloc[i, 0]), remove_special_characters(df.iloc[i, 1])
            self.original.append([s0, s1])
            s0_tokens = word_tokenize(s0)
            s1_tokens = word_tokenize(s1)
            score = df.iloc[i, 2]
            self.processed.append([s0_tokens, s1_tokens, score])
            local_max_len = max(len(s0_tokens), len(s1_tokens))
            if local_max_len > self.max_len:
                self.max_len = local_max_len

    def get_embeddings(self, embedding_type):
        # embedding_types = {"word2vec": getWord2Vec}
        embed = Word2Vec()
        rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2'])
        terp = pd.read_csv(self.terp_path)
        badger = pd.read_csv(self.badger_path)

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
            # BADGER
            additional_features_01.append(badger.iloc[i * 2, 1])
            additional_features_10.append(badger.iloc[i * 2 + 1, 1])
            # sentence lengths
            additional_features_01.append(len(s0))
            additional_features_01.append(len(s1))
            additional_features_01.append(score)
            additional_features_10.append(len(s1))
            additional_features_10.append(len(s0))
            additional_features_10.append(score)

            self.embedded.append([s0_embedded, s1_embedded, additional_features_01])
            self.embedded.append([s1_embedded, s0_embedded, additional_features_10])


class BugRepo(Data):
    def open_and_preprocess(self, path, remove_all):
        print("Removing special chars & stopwords")
        df = pd.read_csv(path)
        if remove_all:
            remove_fn = remove_all_non_alphanumeric
        else:
            remove_fn = remove_special_characters
        for i in range(len(df)):
            s0_original, s1_original = df['s0_title'].iat[i], df['s1_title'].iat[i]
            s0, s1 = remove_fn(s0_original), remove_fn(s1_original)
            s0_tokens = word_tokenize(s0)
            s1_tokens = word_tokenize(s1)
            score = df['score'].iat[i]
            self.processed.append([s0_original, s1_original, s0, s1, s0_tokens, s1_tokens,
                                   len(s0_tokens), len(s1_tokens), score])
            local_max_len = max(len(s0_tokens), len(s1_tokens))
            if local_max_len > self.max_len:
                print('new max length ' + str(local_max_len) + ' at', i)
                self.max_len = local_max_len

    def filter_length(self, min_len, max_len, path):
        processed_df = pd.DataFrame(self.processed, columns=['s0_original', 's1_original', 's0', 's1',
                                                             's0_tokens', 's1_tokens', 's0_len', 's1_len',
                                                             'score'])
        self.max_len = max_len
        self.processed_df_cleaned = processed_df[(processed_df['s0_len'] <= max_len)
                                                 & (processed_df['s0_len'] >= min_len)
                                                 & (processed_df['s1_len'] <= max_len)
                                                 & (processed_df['s1_len'] >= min_len)]
        self.processed_df_cleaned.to_csv(path + 'processed_df_cleaned.csv')

    def prep_for_terp(self, path):
        df = self.processed_df_cleaned
        terp_ref = []
        terp_hyp = []
        for i in range(len(df)):
            s0, s1 = remove_special_characters(df['s0_original'].iat[i]), remove_special_characters(
                df['s1_original'].iat[i])
            annotation = ' ([kle][thunderbird][{index}])'
            terp_ref.append(s0 + annotation.format(index=i))
            terp_hyp.append(s1 + annotation.format(index=i))
        with open(path + "terp.hyp.trans", 'w') as f:
            f.write("\n".join(map(str, terp_hyp)))
        with open(path + "terp.ref.trans", 'w') as f:
            f.write("\n".join(map(str, terp_ref)))

    def prep_for_badger(self, path, mode):
        df = self.processed_df_cleaned
        badger_ref = []
        badger_hyp = []
        for i in range(len(df)):
            s0, s1 = remove_special_characters(df['s0_original'].iat[i]), remove_special_characters(
                df['s1_original'].iat[i])
            annotation = '<seg id=\"{index}\"> '
            badger_ref.append(annotation.format(index=i) + s0 + ' </seg>')
            badger_hyp.append(annotation.format(index=i) + s1 + ' </seg>')
        with open(path + "badger.hyp.sgm", 'w') as f:
            f.write("<tstset trglang=\"en\" setid=\"thunderbird\" srclang=\"any\">")
            f.write("\n<doc sysid=\"" + mode + "\" docid=\"kle\">\n")
            f.write("\n".join(map(str, badger_hyp)))
            f.write("\n</doc>")
            f.write("\n</tstset>")
        with open(path + "badger.ref.sgm", 'w') as f:
            f.write("<refset trglang=\"en\" setid=\"thunderbird\" srclang=\"any\">")
            f.write("\n<doc sysid=\"" + mode + "\" docid=\"kle\">\n")
            f.write("\n".join(map(str, badger_ref)))
            f.write("\n</doc>")
            f.write("\n</refset>")

    def get_embeddings_only(self):
        embed = Word2Vec()

        print("Getting embeddings")
        for i in range(len(self.processed_df_cleaned)):
            s0 = self.processed_df_cleaned['s0_tokens'].iat[i]
            s1 = self.processed_df_cleaned['s1_tokens'].iat[i]
            score = self.processed_df_cleaned['score'].iat[i]
            #             try:
            s0_embedded = np.pad(np.column_stack([embed.get(w) for w in s0]), [[0, 0], [0, self.max_len - len(s0)]],
                                 "constant")
            s1_embedded = np.pad(np.column_stack([embed.get(w) for w in s1]), [[0, 0], [0, self.max_len - len(s1)]],
                                 "constant")
            self.embeddings_only.append([s0_embedded, s1_embedded, score])

    def get_embeddings_and_additional_features(self, terp_path, badger_path):
        embed = Word2Vec()
        rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2'])
        terp = pd.read_csv(terp_path)
        badger = pd.read_csv(badger_path)

        print("Getting embeddings")
        for i in range(len(self.processed_df_cleaned)):
            if (i + 1) % 100 == 0:
                print("Getting embeddings for pair " + str(i + 1) + " out of " + str(len(self.processed) + 1))
            s0_tokens = self.processed_df_cleaned['s0_tokens'].iat[i]
            s1_tokens = self.processed_df_cleaned['s1_tokens'].iat[i]
            s0_original = self.processed_df_cleaned['s0_original'].iat[i]
            s1_original = self.processed_df_cleaned['s1_original'].iat[i]
            score = self.processed_df_cleaned['score'].iat[i]
            s0_embedded = np.pad(np.column_stack([embed.get(w) for w in s0_tokens]),
                                 [[0, 0], [0, self.max_len - len(s0_tokens)]],
                                 "constant")
            s1_embedded = np.pad(np.column_stack([embed.get(w) for w in s1_tokens]),
                                 [[0, 0], [0, self.max_len - len(s1_tokens)]],
                                 "constant")
            # BLEU14_NIST15_METEOR_ROUGE12
            additional_features_01 = extract_mt_features(s0_original, s1_original, rouge)
            # TERp
            additional_features_01.append(terp.iloc[i, 1])
            # BADGER
            additional_features_01.append(badger.iloc[i, 1])
            # sentence lengths
            additional_features_01.append(len(s0_tokens))
            additional_features_01.append(len(s1_tokens))
            additional_features_01.append(score)
            self.embeddings_and_additional_features.append([s0_embedded, s1_embedded, additional_features_01])
