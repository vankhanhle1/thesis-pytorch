import re

import numpy as np
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import single_meteor_score
from nltk.translate.nist_score import sentence_nist


def remove_special_characters(x):
    return re.sub('[$@&<>]', ' ', x)


class StopWordsRemoval():
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))

    def remove_stopwords(self,x):
        tokens = word_tokenize(x)
        x_filtered = [w for w in tokens if not w.lower() in self.stop_words]
        return x_filtered


class Word2Vec():
    def __init__(self):
        # Load Google's pre-trained Word2Vec model.
        self.model = KeyedVectors.load_word2vec_format(
            'C:/Users/leva/Thesis/Embeddings/GoogleNews-vectors-negative300.bin',
            binary=True)
        self.unknowns = np.random.uniform(-0.01, 0.01, 300).astype("float32")

    def get(self, word):
        if word not in self.model.key_to_index:
            word_embedded = self.unknowns
        else:
            word_embedded = self.model.get_vector(word)
        return word_embedded


def extract_mt_features(reference, hypothesis, rouge):
    scores = []
    reference_token = word_tokenize(reference)
    hypothesis_token = word_tokenize(hypothesis)

    bleu_weights = [[1], [1 / 2, 1 / 2], [1 / 3, 1 / 3, 1 / 3], [1 / 4, 1 / 4, 1 / 4, 1 / 4]]
    for weights in bleu_weights:
        scores.append(sentence_bleu(reference_token, hypothesis_token, weights))

    for n_gram in range(1, 6):
        scores.append(sentence_nist(reference_token, hypothesis_token, n_gram))

    scores.append(single_meteor_score(reference, hypothesis))
    rouge_scores = rouge.score(reference, hypothesis)
    for key, values in rouge_scores.items():
        rouge_f1 = list(rouge_scores.get(key))[-1]
        scores.append(rouge_f1)
    return scores
