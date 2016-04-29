#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import glob
import os

import metadata
import preprocessing
#from wikification import DbnlWikifier
import ranking
import feature_extraction
import visualization
import canon


def main():
    """
    #### PREPROCESSING ###################################
    # obtain a dict with the metadata on each text:
    #meta = metadata.metadata_dict()
    # convert the articles in the original xml files to plain text:
    #preprocessing.parse_secondary_dbnl(max_documents=60000)
    # frog the plain articles:
    #preprocessing.frog_articles()
    
    #### WIKIFICATION ###################################
    # construct the wikifier:
    wikifier = DbnlWikifier(workspace='../workspace')
    # collect relevant page_ids for your categories
    wikifier.relevant_page_ids(fresh=False)
    # collect ids of pages that backlink to your relevant pages:
    wikifier.backlinking_pages(fresh=False)
    #backlinks = {k:v for k,v in wikifier.backlinks.items() if len(v) > 100}
    #print(backlinks)
    # collect all mentions of the target pages in the backlinks (this will take a while!)
    wikifier.mentions_from_backlinks(fresh=False)
    # turn the collected mentions into a matrix (vectorization)
    input_dim, output_dim = wikifier.vectorize_wiki_mentions(fresh=False, max_features=2000)
    # optimize a classifier to label new mentions:
    #dev_acc, test_acc = wikifier.classifier(input_dim=input_dim, output_dim=output_dim,
    #                                        fresh=True, test=True, nb_epochs=200,
    #                                         hidden_dim=1024)
    # train the final classifier on all data:
    #wikifier.classifier(fresh=True, input_dim=input_dim, output_dim=output_dim,
    #                    test=False, nb_epochs=50,
    #                    hidden_dim=1024)
    
    ######## (the following is specific to the dbnl data) #######################################################
    # collect all unique NEs in the corpus
    # and get the pages which the wikipedia search interface links them to:
    #testfiles = glob.glob('../workspace/corr_rnd_sample/*.wikified')
    #testfiles = [os.path.basename(tf).replace('.wikified', '') for tf in testfiles]
    #wikifier.extract_unique_nes(fresh=False, max_documents=1000000000, max_words_per_doc=10000000000000, testfiles=testfiles)
    wikifier.extract_unique_nes(fresh=False, max_documents=1000000000, max_words_per_doc=10000000000000)
    # use the trained wikifier to disambiguate the NEs in the corpus:
    #wikifier.disambiguate_nes(max_documents=50000, max_words_per_doc=10000000, testfiles=testfiles)
    #wikifier.naive_disambiguate_nes(max_documents=100000000000, max_words_per_doc=10000000000000, testfiles=testfiles)
    wikifier.naive_disambiguate_nes(max_documents=100000000000, max_words_per_doc=10000000000000)
    #wikifier.evaluate_sample()
    
    """

    # EXTRACT FEATURES
    #feature_extraction.extract_features(max_documents=1000,
    #                   max_words_per_doc=500,
    #                   incl_tf=True,
    #                   incl_df=True,
    #                   incl_graph=True,
    #                   incl_w2v=True,
    #                   incl_topic_model=True,
    #                   incl_atm=True)

    r = ranking.Ranker()
    C = canon.get_canon('mnl')

    # SIMPLEX
    rr = r.simplex_rankings()
    for rr, rank in rr.items():
        print('=======')
        print(rr)
        print(rank)
        print('overlap:', canon.overlap_score(correct_canon=C,
                                              predicted_canon=rank))
        print('tau:', canon.kendall_tau(correct_canon=C,
                                        predicted_canon=rank))
        print('mrr:', canon.mean_reciprocal_rank(correct_canon=C,
                                        predicted_canon=rank))

    # ATM
    #for rr, rank in r.atm_single_topic_ranking().items():
    #    print('=======')
    #    print(rr)
    #    print(rank)

    # W2V
    #for rr, rank in r.w2v_ranking().items():
    #    print('=======')
    #    print(rr)
    #    print(rank)


    """
    kantl = canon.get_canon('kantl')
    print('overlap:', canon.overlap_score(correct_canon=kantl, predicted_canon=mf))
    print('tau:', canon.kendall_tau(correct_canon=kantl, predicted_canon=mf))
    print('mrr:', canon.mean_reciprocal_rank(correct_canon=kantl, predicted_canon=mf))
    mf = [k for k, _ in self.df.most_common()][:50]
    print('overlap:', canon.overlap_score(correct_canon=kantl, predicted_canon=mf))
    print('tau:', canon.kendall_tau(correct_canon=kantl, predicted_canon=mf))
    print('mrr:', canon.mean_reciprocal_rank(correct_canon=kantl, predicted_canon=mf))
    """


    # SOME FEATURE VISUALIZATIONS
    #visualization.author_frequency_barplot()
    #visualization.w2v_author_similarities()
    #visualization.author_tsne_plot()
    #visualization.temporal_lda_plot(max_documents=1000,
    #                  max_words_per_doc=100,
    #                  n_topics=100)

    # SOME METADATA VISUALIZATIONS
    #actual_texts = []
    #for filepath in glob.glob('../texts/*.xml'):
    #    text_id = os.path.splitext(os.path.basename(filepath))[0][:-3] # remove trailing "_01"
    #    actual_texts.append(text_id)
    #d = metadata.metadata_dict()
    #visualization.plot_nb_texts(d, actual_texts)
    #visualization.plot_size_texts(d)
    #visualization.plot_periodicals(d)
    #visualization.plot_frogged_data(d)

if __name__ == '__main__':
    main()