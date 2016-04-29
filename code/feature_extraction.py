from __future__ import print_function

import glob
import codecs
import re
import os
import sys
from collections import Counter, defaultdict, OrderedDict
from itertools import combinations
from operator import itemgetter
import pandas as pd
import pickle

import networkx as nx
import numpy as np
from scipy import stats

from gensim.models import Word2Vec
from gensim.models.wrappers import LdaMallet
from gensim import corpora

from ptm import AuthorTopicModel
from ptm.utils import convert_cnt_to_list, get_top_words


class DirectoryIterator:
    def __init__(self, path_pattern, max_documents, max_words_per_doc,
                 get='wiki', lda_dict=None):
        self.path_pattern = path_pattern
        self.max_documents = max_documents
        self.max_words_per_doc = max_words_per_doc
        self.get = get
        self.lda_dict = lda_dict

    def __iter__(self):
        max_documents = self.max_documents
        for filename in sorted(glob.glob(self.path_pattern)):
            words = []
            max_documents -= 1
            if max_documents % 100 == 0:
                print('\t-', max_documents, 'to go')
            if max_documents <= 1:
                break
            if self.get == 'filename':
                yield filename
            else:
                word_cnt = self.max_words_per_doc
                for line in codecs.open(filename, 'r', encoding='utf8'):
                    comps = line.strip().split('\t')
                    if comps:
                        idx, token, lemma, pos, pos_conf, ner, wiki = comps
                        if self.get == 'wiki':
                            if wiki != 'X':
                                words.append(wiki)
                        elif self.get == 'w2v':
                            if wiki != 'X':
                                words.append('*'+wiki)
                            elif ner == 'O':# and pos.startswith(('N(', 'ADJ(', 'WW(')):
                                words.append(token.lower())
                        elif self.get in ('lda', 'lda_vocab') and pos.startswith(('N(', 'ADJ(')):
                            words.append(token.lower())
                    word_cnt -= 1
                    if word_cnt <= 0:
                        break
                if self.get == 'lda':
                    yield self.lda_dict.doc2bow(words)
                else:
                    yield words

def extract_features(max_documents=50000000,
                     max_words_per_doc=50000000,
                     incl_tf=True,
                     incl_df=True,
                     incl_graph=True,
                     incl_w2v=True,
                     incl_topic_model=True,
                     incl_atm=True):
    
    ######### SIMPLE FREQUENCY MEASURES ######################################################
    if incl_df or incl_tf or incl_graph:
        doc_cnt = max_documents
        # set containers:
        tf, df, network = Counter(), Counter(), nx.Graph()
        doc_ner_idx = {}
        dir_ner_iterator = DirectoryIterator(path_pattern='../workspace/wikified_periodicals/*.wikified',
                                         max_documents=max_documents,
                                         max_words_per_doc=max_words_per_doc,
                                         get='wiki')
        dir_filename_iterator = DirectoryIterator(path_pattern='../workspace/wikified_periodicals/*.wikified',
                                         max_documents=max_documents,
                                         max_words_per_doc=max_words_per_doc,
                                         get='filename')
        for filename, words in zip(dir_filename_iterator, dir_ner_iterator):
            # count the ners:
            ner_cnt = Counter()
            ner_cnt.update(words)
            if ner_cnt:
                # collect which ners appear in which doc:
                doc_ner_idx[os.path.basename(filename)] = set([n for n in ner_cnt])
                # update global tf and df:
                for k, v in ner_cnt.items():
                    tf[k] += v
                    df[k] += 1
                # update nodes in network:
                for ner in ner_cnt:
                    if ner not in network:
                        network.add_node(ner)
                # update edges in network:
                for ner1, ner2 in combinations(ner_cnt, 2):
                    try:
                        network[ner1][ner2]['weight'] += 1
                    except KeyError:
                        network.add_edge(ner1, ner2, weight=1)
        
        # dump for reuse:
        pickle.dump(tf, open('../workspace/tf.m', 'wb'))
        pickle.dump(df, open('../workspace/df.m', 'wb'))
        pickle.dump(doc_ner_idx, open('../workspace/doc_ner_idx.m', 'wb'))
        pickle.dump(network, open('../workspace/nx.m', 'wb'))
        
        # scale network values:
        max_weight = float(max([network[n1][n2]['weight']\
                            for n1, n2 in network.edges_iter()]))
        for n1, n2 in network.edges_iter():
            network[n1][n2]['weight'] /= max_weight
        nx.write_gexf(network,
                      '../workspace/dbnl_network.gexf',
                      prettyprint=True)
    
    ######### WORD2VEC MODEL ######################################################
    if incl_w2v:
        # build w2v model:
        dir_w2v_iterator = DirectoryIterator(path_pattern='../workspace/wikified_periodicals/*.wikified',
                                         max_documents=max_documents,
                                         max_words_per_doc=max_words_per_doc,
                                         get='w2v')
        w2v_model = Word2Vec(dir_w2v_iterator, window=15, min_count=10,
                                         size=150, workers=10, negative=5)
        w2v_model.init_sims(replace=True)
        w2v_model.save(os.path.abspath('../workspace/w2v_model.m'))

    ######### STANDARD TOPIC MODEL ######################################################
    if incl_topic_model:
        # build vocab for lda:
        vocab_lda_iterator = DirectoryIterator(path_pattern='../workspace/wikified_periodicals/*.wikified',
                                         max_documents=max_documents,
                                         max_words_per_doc=max_words_per_doc,
                                         get='lda_vocab')
        lda_dict = corpora.Dictionary(vocab_lda_iterator)
        lda_dict.filter_extremes(no_below=25, no_above=0.5, keep_n=5000)
        
        # build lda model:
        dir_lda_iterator = DirectoryIterator(path_pattern='../workspace/wikified_periodicals/*.wikified',
                                         max_documents=max_documents,
                                         max_words_per_doc=max_words_per_doc,
                                         get='lda',
                                         lda_dict=lda_dict)
        lda_workspace_path = '../workspace/mallet_output/'
        if not os.path.isdir(lda_workspace_path):
            os.mkdir(lda_workspace_path)
        mallet_path = '/home/mike/GitRepos/dbnl/code/mallet-2.0.8RC2/bin/mallet'
        lda_model = LdaMallet(mallet_path, dir_lda_iterator, num_topics=150,
                                       id2word=lda_dict, iterations=1900,
                                       prefix=lda_workspace_path)
        lda_model.save('../workspace/lda_model.m')

    ######### AUTHOR TOPIC MODEL ######################################################
    if incl_atm:
        # build vocab for lda:
        vocab_lda_iterator = DirectoryIterator(path_pattern='../workspace/wikified_periodicals/*.wikified',
                                         max_documents=max_documents,
                                         max_words_per_doc=max_words_per_doc,
                                         get='lda_vocab')
        lda_dict = corpora.Dictionary(vocab_lda_iterator)
        lda_dict.filter_extremes(no_below=25, no_above=0.5, keep_n=5000)
        lda_dict.compactify()
        atm_vocab = []
        for i, w in lda_dict.items():
            atm_vocab.append(w)
        print(len(atm_vocab), 'vocab')
        atm_vocab = tuple(atm_vocab)
        corpus, doc_author = [], []
        for filename in sorted(glob.glob('../workspace/wikified_periodicals/*.wikified')):
            doc_words, auth_set = [], set()
            max_documents -= 1
            if max_documents % 100 == 0:
                print('\t-', max_documents, 'to go')
            if max_documents <= 1:
                break
            word_cnt = max_words_per_doc
            for line in codecs.open(filename, 'r', encoding='utf8'):
                comps = line.strip().split('\t')
                if comps:
                    idx, token, lemma, pos, pos_conf, ner, wiki = comps
                    if wiki != 'X':
                        auth_set.add(wiki)
                    elif pos.startswith(('N(', 'ADJ(')):
                        try:
                            doc_words.append(atm_vocab.index(token.lower()))
                        except:
                            pass
                word_cnt -= 1
                if word_cnt <= 0:
                    break
            if auth_set and doc_words:
                corpus.append(sorted(doc_words))
                doc_author.append(sorted(list(auth_set)))
        atm_author_idx = {}
        for i1, authors in enumerate(doc_author):
            for i2, auth in enumerate(authors):
                if auth not in atm_author_idx:
                    atm_author_idx[auth] = len(atm_author_idx)
                doc_author[i1][i2] = atm_author_idx[auth]
        n_topic = 30
        atm_model = AuthorTopicModel(n_doc=len(corpus),
                                     n_voca=len(atm_vocab),
                                     n_topic=n_topic,
                                     n_author=len(atm_author_idx))
        atm_model.fit(corpus, doc_author, max_iter=10)
        for k in range(n_topic):
            top_words = get_top_words(atm_model.TW, atm_vocab, k, 10)
            print('topic ', k , ','.join(top_words))
        author_id = 7
        fig = plt.figure(figsize=(12,6))
        plt.bar(range(n_topic), atm_model.AT[author_id]/np.sum(atm_model.AT[author_id]))
        #plt.title(author_idx[author_id])
        plt.xticks(np.arange(n_topic)+0.5, ['\n'.join(get_top_words(atm_model.TW, atm_vocab, k, 10)) for k in range(n_topic)])
        #plt.show()
        plt.savefig('atm1.pdf')
        pickle.dump(atm_vocab, open('../workspace/atm_vocab.m', 'wb'))
        pickle.dump(atm_model, open('../workspace/atm_model.m', 'wb'))
        pickle.dump(atm_author_idx, open('../workspace/atm_author_idx.m', 'wb'))

        

        

