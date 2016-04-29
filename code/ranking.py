import os
import pickle
from operator import itemgetter

from scipy import stats
import numpy as np
import networkx as nx

from gensim.models import Word2Vec

class Ranker:
    def __init__(self):
        self.atm_model = pickle.load(open('../workspace/atm_model.m', 'rb'))
        self.atm_vocab = pickle.load(open('../workspace/atm_vocab.m', 'rb'))
        self.atm_author_idx = pickle.load(open('../workspace/atm_author_idx.m', 'rb'))
        self.network = pickle.load(open('../workspace/nx.m', 'rb'))
        self.w2v_model = Word2Vec.load(os.path.abspath('../workspace/w2v_model.m'))
        self.tf = pickle.load(open('../workspace/tf.m', 'rb'))
        self.df = pickle.load(open('../workspace/df.m', 'rb'))
        self.doc_ner_idx = pickle.load(open('../workspace/doc_ner_idx.m', 'rb'))
        self.network = pickle.load(open('../workspace/nx.m', 'rb'))

    def simplex_rankings(self, nb=50):
        rankings = {}
        rankings['tf'] = self.tf_rank(nb=nb)
        rankings['df'] = self.df_rank(nb=nb)
        rankings['tau'] = self.tau_rank(nb=nb)
        rankings['page (no weights)'] = self.page_rank(nb=nb, with_weights=False)
        rankings['page (with weights)'] = self.page_rank(nb=nb, with_weights=True)
        rankings['between (no weights)'] = self.betweenness_centrality_rank(nb=nb, with_weights=False)
        rankings['degree'] = self.degree_centrality_rank(nb=nb)
        rankings['closeness'] = self.closeness_centrality_rank(nb=nb)
        rankings['atm_cv'] = self.atm_cv_rank(nb=nb)
        return rankings

    def tf_rank(self, nb=50):
        return [k for k, _ in self.tf.most_common()][:nb]

    def df_rank(self, nb=50):
        return [k for k, _ in self.df.most_common()][:nb]

    def tau_rank(self, nb=50):
        # growth TAU part
        years = {str(k):0 for k in range(1945, 2015)}
        diachro = {}
        for k in list(self.doc_ner_idx.keys())[:100]:
            auths = self.doc_ner_idx[k]
            year = k.split('-')[-1].replace('.wikified', '')
            for auth in auths:
                if auth not in diachro:
                    diachro[auth] = years.copy()
                diachro[auth][year] += 1

        tau_dict = {}
        for auth, cnt_dict in diachro.items():
            h = list(range(1945, 2015))
            cnts = [cnt_dict[str(y)] for y in h]
            corr, pval = stats.kendalltau(cnts, h)
            if pval > 0.05:
                tau_dict[auth] = corr
        suggested_list = sorted(tau_dict.items(), key=itemgetter(1), reverse=True)
        return [k for k, v in suggested_list][:nb]

    def atm_cv_rank(self, nb=50):
        cv_dict = {}
        for auth, auth_id in self.atm_author_idx.items():
            scores = self.atm_model.AT[auth_id] / np.sum(self.atm_model.AT[auth_id])
            cv_dict[auth] = stats.variation(scores)
        suggested_list = sorted(cv_dict.items(), key=itemgetter(1)) # don't reverse this!
        return [k for k, v in suggested_list][:nb]

    def page_rank(self, with_weights=True, nb=50):
        if with_weights:
            centrality = nx.pagerank(self.network)
        else:
            centrality = nx.pagerank(self.network, weight=None)
        suggested_authors = []
        for name, _ in sorted(centrality.items(), key=itemgetter(1), reverse=True):
            suggested_authors.append(name.replace('*', ''))
        return suggested_authors[:nb]

    def degree_centrality_rank(self, nb=50):
        centrality = nx.degree_centrality(self.network)
        suggested_authors = []
        for name, _ in sorted(centrality.items(), key=itemgetter(1), reverse=True):
            suggested_authors.append(name.replace('*', ''))
        return suggested_authors[:nb]

    def closeness_centrality_rank(self, nb=50):
        centrality = nx.closeness_centrality(self.network)
        suggested_authors = []
        for name, _ in sorted(centrality.items(), key=itemgetter(1), reverse=True):
            suggested_authors.append(name.replace('*', ''))
        return suggested_authors[:nb]

    def betweenness_centrality_rank(self, with_weights=True, nb=50):
        if with_weights:
            centrality = nx.betweenness_centrality(self.network)
        else:
            centrality = nx.betweenness_centrality(self.network, weight=None)
        suggested_authors = []
        for name, _ in sorted(centrality.items(), key=itemgetter(1), reverse=True):
            suggested_authors.append(name.replace('*', ''))
        return suggested_authors[:nb]

    def w2v_ranking(self, nb=50):
        rankings = {}
        vocab = list(self.w2v_model.vocab)
        for w in vocab:
            scores = []
            for auth in self.atm_author_idx:
                if '*'+auth in vocab:
                    scores.append((auth, self.w2v_model.similarity('*'+auth, w)))
            suggested_authors = sorted(scores, key=itemgetter(1), reverse=True) # reverse, since similarity
            rankings[w] = suggested_authors
        return rankings

    def atm_single_topic_ranking(self, nb=50):
        rankings = {}
        for topic_idx in range(self.atm_model.AT.shape[1]):
            scores = self.atm_model.AT.copy()
            scores /= scores.sum(axis=1)[:, np.newaxis]
            indiv_scores = []
            for auth, auth_idx in self.atm_author_idx.items():
                indiv_scores.append((auth, scores[auth_idx, topic_idx]))
            suggested_list = sorted(indiv_scores, key=itemgetter(1), reverse=True)
            suggested_list = [k for k, v in suggested_list][:nb]
            rankings[topic_idx] = suggested_list
        return rankings

    
