from __future__ import print_function

import glob
import codecs
import re
import os
from collections import Counter, defaultdict
from itertools import combinations
import cPickle as pickle

import networkx as nx
from gensim.models import Word2Vec
from gensim.models.wrappers import LdaMallet
from gensim import corpora



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
                        elif pos.startswith(('N(', 'ADJ(', 'WW(')):
                            words.append(token.lower())
                    elif self.get in ('lda', 'lda_vocab') and pos.startswith(('N(', 'ADJ(', 'WW(')):
                        words.append(token.lower())
                word_cnt -= 1
                if word_cnt <= 0:
                    break
            if self.get == 'lda':
                yield self.lda_dict.doc2bow(words)
            elif self.get == 'filename':
                yield filename
            else:
                yield words

class Featurizer:
    def __init__(self,
                 incl_tf=True,
                 incl_df=True,
                 incl_doc_cooccur=True,
                 incl_w2v=True,
                 incl_temporal=True,
                 incl_topic_model=True):

        self.incl_tf = incl_tf
        self.incl_df = incl_df
        self.incl_doc_cooccur = incl_doc_cooccur
        self.incl_w2v = incl_w2v
        self.incl_topic_model = incl_topic_model
        self.incl_temporal = incl_temporal

    def featurize(self, max_documents=5000, max_words_per_doc=5000):
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

        print(tf.most_common(30))
        print(df.most_common(30))
        # dump for reuse:
        pickle.dump(tf, open('../workspace/tf.m', 'wb'))
        pickle.dump(tf, open('../workspace/df.m', 'wb'))
        pickle.dump(tf, open('../workspace/ners_per_doc.m', 'wb'))
        
        # build w2v model:
        dir_w2v_iterator = DirectoryIterator(path_pattern='../workspace/wikified_periodicals/*.wikified',
                                         max_documents=max_documents,
                                         max_words_per_doc=max_words_per_doc,
                                         get='w2v')
        w2v_model = Word2Vec(dir_w2v_iterator, window=15, min_count=1,
                                         size=300, workers=1, negative=5)
        w2v_model.init_sims(replace=True)
        w2v_model.init_sims(replace=True)
        w2v_model.save(os.path.abspath('../workspace/w2v_model.m'))
        

        # build vocab for lda:
        vocab_lda_iterator = DirectoryIterator(path_pattern='../workspace/wikified_periodicals/*.wikified',
                                         max_documents=max_documents,
                                         max_words_per_doc=max_words_per_doc,
                                         get='lda_vocab')
        lda_dict = corpora.Dictionary(vocab_lda_iterator)
        lda_dict.filter_extremes(no_below=25, no_above=0.5, keep_n=10000)

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
        lda_model = LdaMallet(mallet_path, dir_lda_iterator, num_topics=100,
                                       id2word=lda_dict, iterations=100,
                                       prefix=lda_workspace_path)
        lda_model.save('../workspace/lda_model.m')

#tf = pickle.load(open('../workspace/tf.m', "rb"))
#df = pickle.load(open('../workspace/df.m', "rb"))
#w2v_model = Word2Vec.load(os.path.abspath('../workspace/w2v_model.m'))
        

