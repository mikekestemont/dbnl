#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import metadata
import preprocessing
from wikification import Wikifier

def main():
    #### PREPROCESSING ###################################
    # obtain a dict with the metadata on each text:
    meta = metadata.metadata_dict()
    # convert the articles in the original xml files to plain text:
    preprocessing.parse_secondary_dbnl(max_documents=60000)
    # frog the plain articles:
    #preprocessing.frog_articles()
    """
    #### WIKIFICATION ###################################
    # construct the wikifier:
    wikifier = Wikifier()
    # collect relevant page_ids for your categories
    wikifier.relevant_page_ids(fresh=False)
    page_ids = wikifier.page_ids[:100]
    # collect ids of pages that backlink to your relevant pages:
    wikifier.backlinking_pages(page_ids=page_ids, fresh=False)
    # collect all mentions of the target pages in the backlinks (this will take a while!)
    wikifier.mentions_from_backlinks(fresh=False)
    # turn the collected mentions into a matrix (vectorization)
    input_dim, output_dim = wikifier.vectorize_wiki_mentions(fresh=False, max_features=500)
    # optimize a classifier to label new mentions:
    #dev_acc, test_acc = wikifier.classifier(input_dim=input_dim, output_dim=output_dim, fresh=True, test=True, nb_epochs=2)
    # train the final classifier on all data:
    wikifier.classifier(input_dim=input_dim, output_dim=output_dim, fresh=False, test=False, nb_epochs=1)
    ######## (the following is specific to the dbnl data) #######################################################
    # collect all unique NEs in the corpus
    # and get the pages which the wikipedia search interface links them to:
    wikifier.extract_unique_nes(fresh=False, max_documents=100, max_words_per_doc=150)
    # use the trained wikifier to disambiguate the NEs in the corpus:
    wikifier.disambiguate_nes(max_documents=100, max_words_per_doc=150)
    """
    """
    Feature types to implement:
        - network cooccurence features
        - topic model features > similarity to other authors, similarity to keywords such as 'roman', 'gedicht', ...
        - word2vec features
        - frequency (also coefficient of variation)
        - sentiment features (both +/- and extremity)
        - temporal features: e.g. frequency per decade
        - wikipedia related features: e.g. pagerank on wikipedia, categorie on wikipedia (e.g. period?) > attention: to which extend interrelated?
    """

if __name__ == '__main__':
    main()