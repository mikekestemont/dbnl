#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import pickle
import re
import codecs
import glob
import logging
import os
import shutil
import tempfile
from collections import Counter
from xml.dom import minidom
from xml.dom.minidom import Node
from operator import itemgetter

from bs4 import BeautifulSoup
from scipy.sparse import hstack

from pattern.web import Wikipedia, plaintext

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.metrics import f1_score

from utils import query, _plaintext, neural_model

from keras.utils import np_utils


logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)


class Wikifier(object):

    def __init__(self, workspace=None, relevant_categories=None):
        """
        Intialize wikifier with the relevant categories you wish to extract.
        """
        self.relevant_categories = relevant_categories
        # make sure that we have a workspace where we can pickle:
        self.workspace = workspace
        if not os.path.isdir(self.workspace):
            os.path.mkdir(self.workspace)

    def relevant_page_ids(self, fresh=False, filename='relevant_page_ids.p'):
        """
        Collect the wiki id's (page_ids) for all pages belonging to the self.relevant_categories
            * if fresh: fresh download from Wikipedia + pickled under filename
            * else: no download; page_ids loaded from pickle in filename
        """
        logging.info('Getting page_ids from relevant categories.')
        self.page_ids = set()

        if fresh:
            for category in self.relevant_categories:
                for result in query({'generator': 'categorymembers', 'gcmtitle': 'Category:'+category, 'gcmlimit': '500'}):
                    for page in result['pages']:
                        page_id = result['pages'][page]['title']
                        self.page_ids.add(page_id)
                        if len(self.page_ids) % 1000 == 0:
                            logging.info(
                                '\t+ nb pages download: %d' % len(self.page_ids))

            self.page_ids = sorted(self.page_ids)
            with open(os.path.join(self.workspace, filename), 'wb') as out:
                pickle.dump(self.page_ids, out)

        else:
            with open(os.path.join(self.workspace, filename), 'rb') as inf:
                self.page_ids = pickle.load(inf)

        logging.info('\t+ set %d page_ids.' % len(self.page_ids))

        return self.page_ids

    def backlinking_pages(self, page_ids=None, ignore_categories=None, fresh=False, filename='backlinks.p'):
        """
        Sets a dict (backlinks), with for each page_id a set of the pages backlinking to it.
            * if fresh: fresh download + pickle under outfilename
            * else: no download; backlinks loaded from pickle in filename
        In the case of new download:
            * if page_ids = None, self.page_ids is used
            * categories starting with one of the items in ignore_categories will be ignored
        """
        self.backlinks = {}

        if fresh:
            if not page_ids:
                page_ids = self.page_ids
            logging.info('Collecting backlinks for %d pages' % len(page_ids))

            if not ignore_categories:
                ignore_categories = (
                    'Gebruiker:', 'Lijst van', 'Portaal:', 'Overleg', 'Wikipedia:', 'Help:', 'Categorie:')

            for idx, page_id in enumerate(page_ids):
                self.backlinks[page_id] = set()
                for result in query({'action': 'query', 'list': 'backlinks', 'format': 'json', 'bltitle': page_id}):
                    for backlink in result['backlinks']:
                        backlink = backlink['title'].replace(
                            '_', ' ')  # clean up
                        if not backlink.startswith(ignore_categories):
                            self.backlinks[page_id].add(backlink)
                if idx % 10 == 0:
                    logging.info('\t+ collected %d backlinks for %d pages' % (
                        sum([len(v) for k, v in self.backlinks.items()]), idx+1))

            # remove pages without relevant backlinks
            self.backlinks = {k: v for k, v in self.backlinks.items() if v}
            # dump for later reuse
            with open(os.path.join(self.workspace, filename), 'wb') as out:
                pickle.dump(self.backlinks, out)
        
        else:
            with open(os.path.join(self.workspace, filename), 'rb') as inf:
                self.backlinks = pickle.load(inf)

        logging.info('\t+ loaded %d backlinks for %d pages' % (
            sum([len(v) for k, v in self.backlinks.items()]), len(self.backlinks)))

    def mentions_from_backlinks(self, backlinks={}, fresh=False, filename='mentions.p', context_window_size=150):
        """
        Mines backlinking pages for mentions of the page_ids in backlinks.
        Returns 5 tuples, with for each mention:
            * target_id (correct page title)
            * the name variant (inside the a-tag)
            * left context of the mention (continguous character string, with len = context_window_size)
            * right context of the mention (continguous character string, with len = context_window_size)
            * a counter of other page_ids mentioned on the page
        """
        if not backlinks:
            backlinks = self.backlinks

        # intitialize data containers:
        target_ids, name_variants, left_contexts, right_contexts, page_counts = [], [], [], [], []

        if fresh:

            logging.info('Mining mentions from %d backlinking pages to %d target pages.' % (
                sum([len(v) for k, v in backlinks.items()]), len(backlinks)))

            wikipedia = Wikipedia(language='nl', throttle=2)

            for idx, (page_id, links) in enumerate(backlinks.items()):

                logging.debug('\t + mining mentions of %s (%s backlinks) | page %d / %d' % (
                    page_id, len(links), idx+1, len(backlinks)))

                for backlink in links:
                    article = wikipedia.search(backlink)
                    # skip referral pages
                    if article and not article.categories[0] == 'Wikipedia:Doorverwijspagina':
                        logging.debug('\t\t* backlink: %s' % backlink)
                        # fetch the html-sections of individual sections:
                        section_sources = []
                        # article doesn't have sections
                        if not article.sections:
                            section_sources = [article.source]
                        else:
                            section_sources = [section.source for section in article.sections]
                        # loop over the section sources and extract all
                        # relevant mentions:
                        for section_source in section_sources:
                            ts, nvs, lcs, rcs, cnts = self.mentions_from_section(source=section_source,
                                                                                 target_id=page_id,
                                                                                 context_window_size=context_window_size)
                            if nvs:
                                target_ids.extend(ts)
                                name_variants.extend(nvs)
                                left_contexts.extend(lcs)
                                right_contexts.extend(rcs)
                                page_counts.extend(cnts)

            with open(os.path.join(self.workspace, filename), 'wb') as out:
                pickle.dump((target_ids, name_variants, left_contexts,
                             right_contexts, page_counts), out)

        else:
            with open(os.path.join(self.workspace, filename), 'rb') as inf:
                target_ids, name_variants, left_contexts, right_contexts, page_counts = pickle.load(inf)

        self.mentions = (target_ids, name_variants, left_contexts, right_contexts, page_counts)

    def featurize_section(self, formatted, page_cnt, context_window_size):
        """
        Takes a string in which author NEs have been tagged as <author>variant</author>
        Returns tuples with for each author: target_id, name_variant, left_context, right_context, page_count
        Ignores author-elements that have an attribute type=unambiguous
        """
        loc_target_labels, loc_name_variants, loc_left_contexts, loc_right_contexts, loc_page_counts = [], [], [], [], []

        # keep minimum of chars (to avoid encoding errors)
        formatted = "".join([char for char in formatted if
                             (char.isalpha() or char.isdigit() or char.isspace() or char in '<>/=_.-,"()%#' or char == "'")])
        # add root tags for parsability:
        formatted = '<root>'+formatted+'</root>'
        # parse the xml representation of the section:
        try:
            tree = minidom.parseString(formatted.encode('utf-8'))
        except:
            return None

        for node in tree.getElementsByTagName('author'):
            if 'type' in node.attributes.keys() and node.attributes['type'].value == 'unambiguous':
                # skip these during testing, where we don't disambiguate all
                # NEs:
                continue

            # first cycle through left context and collect sufficient chars:
            left_context, prev_node = '', None
            while len(left_context) < context_window_size:
                try:
                    if prev_node == None:
                        prev_node = node.previousSibling
                    else:
                        prev_node = prev_node.previousSibling
                    if prev_node.nodeType == Node.TEXT_NODE:
                        left_context = prev_node.data + \
                            left_context  # mind the order!
                    else:
                        left_context = prev_node.firstSibling.data + \
                            left_context  # mind the order!
                except AttributeError:  # end of tree (no next)
                    break
            left_context = ' '.join(
                left_context.strip().split())[-context_window_size:]  # cut

            # now, parallel cycle through right context:
            right_context, next_node = '', None
            while len(right_context) < context_window_size:
                try:
                    if next_node == None:
                        next_node = node.nextSibling
                    else:
                        next_node = next_node.nextSibling
                    if next_node.nodeType == Node.TEXT_NODE:
                        right_context += next_node.data
                    else:
                        right_context += next_node.firstSibling.data
                except AttributeError:  # beginning of tree (no next)
                    break
            right_context = ' '.join(
                right_context.strip().split())[:context_window_size]  # cut

            try:
                # append to containers:
                # extract variant name; mark beginning and ending of strings
                loc_name_variants.append("%"+node.firstChild.data+"$")
                if 'id' in node.attributes.keys():
                    # extract class label, if available
                    l = node.attributes['id'].value
                    if "#" in l: # catch in-page referral
                        l = l.split('#')[0]
                    loc_target_labels.append(l)
                else:
                    loc_target_labels.append('X')
                loc_left_contexts.append(left_context)
                loc_right_contexts.append(right_context)
                # note that this cntr is the same for all mentions in the same
                # section
                loc_page_counts.append(page_cnt)
            except AttributeError:
                pass
        if loc_name_variants:
            len_test = len(loc_name_variants)
            if len_test and len_test*5 == sum((len(loc_target_labels), len(loc_name_variants), len(loc_left_contexts),\
                                              len(loc_right_contexts), len(loc_page_counts))):
                return loc_target_labels, loc_name_variants, loc_left_contexts, loc_right_contexts, loc_page_counts
        return None

    def mentions_from_section(self, source, target_id, context_window_size=150):
        """
        Takes the html source of a wikipedia section and returns 5 tuples, with for each link to target_id:
            * target_id (wikipedia id of the target page)
            * the name variant (inside the a-tag)
            * left context of the mention (contiguous character string, with len = context_window_size)
            * right context of the mention (contiguous character string, with len = context_window_size)
            * counts of other mentions of self.page_ids in the section (should we do this at the page level?)
        Note that other mentions of relevant pages occuring in the context are
        included as written (i.e. not the normalized wiki-id).
        """
        target_labels, name_variants, left_contexts, right_contexts, page_counts = [], [], [], [], []

        # count other mentions of pages in self.page_ids in this section (for
        # global disambiguation):
        page_cnt = Counter()
        # add tags for parsability
        soup = BeautifulSoup('<html>'+source+'</html>', 'html.parser')
        for a_node in soup.find_all('a'):  # iterate over all hyperlinks
            try:
                link, title = a_node.get('href'), a_node.get(
                    'title').replace('_', ' ')
                # check whether the link is of interest
                if link.startswith('/wiki/') and title in self.page_ids:
                    if '#' in title:
                        title = title.split("#")[0]
                    page_cnt[title] += 1
            except AttributeError:
                pass
        page_cnt = {k: (v/sum(page_cnt.values(), 0.0))
                    for k, v in page_cnt.items()}  # normalize absolute counts

        # convert html to plain text, but preserve links (a-elements):
        clean_text_with_links = plaintext(
            html=source, keep={'a': ['title', 'href']})
        # reformat relevant author links to tmp syntax (%Hugo_Claus|Claus%):
        tmp_id = target_id.replace('(', '\(').replace(')', '\)').replace('.', '\.')#.replace('%', '\%')
        author_pattern = re.compile(
            r'<a href="/wiki/([^\"]+)" title="'+tmp_id+r'\">([^\<]+)</a>')
        formatted = author_pattern.sub(
            repl=r'@\1|\2@', string=clean_text_with_links)
        # now remove remaining, irrelevant links (e.g. external links):
        formatted = _plaintext(formatted)
        # and reformat into xml:
        # correct for irregularities (too long a name variant)
        author_pattern = re.compile(r'\@([^\@]{0,50})\|([^\@]{0,50})\@')
        formatted = author_pattern.sub(
            repl=r'<author id="\1">\2</author>', string=formatted)
        featurized = self.featurize_section(
            formatted, page_cnt, context_window_size)
        if featurized:
            loc_target_labels, loc_name_variants, loc_left_contexts, loc_right_contexts, loc_page_counts = featurized
            if loc_name_variants:
                # append to containers:
                target_labels.extend(loc_target_labels)
                name_variants.extend(loc_name_variants)
                left_contexts.extend(loc_left_contexts)
                right_contexts.extend(loc_right_contexts)
                page_counts.extend(loc_page_counts)

        return target_labels, name_variants, left_contexts, right_contexts, page_counts

    def vectorize_dbnl_nes(self, mentions):
        """
        Vectorize new unseen strings in which mentions using the previously fitted vectorizers
        """
        _, name_variants, left_contexts, right_contexts, page_counts = mentions
        variant_vecs = self.variant_vectorizer.transform(name_variants)
        left_context_vecs = self.context_vectorizer.transform(left_contexts)
        right_context_vecs = self.context_vectorizer.transform(right_contexts)
        cnt_vecs = self.page_cnt_vectorizer.transform(page_counts)
        # concatenate all matrices:
        X = hstack((variant_vecs, left_context_vecs, right_context_vecs, cnt_vecs))
        #X = variant_vecs
        return X  # still sparse!

    def classify_nes(self, X, original_tokens):
        """
        Takes the vectorized matrix X for a list of new tokens and returns the disambiguated pages
        """
        winners = []
        predictions = self.model.predict(X.toarray())  # unsparsify
        # rm special symbols surrounding strings
        original_tokens = [t[1:-1] for t in original_tokens]
        for prediction, token in zip(predictions, original_tokens):
            # only consider class labels returned by the wikipedia search for
            # token:
            options = [o for o in self.nes2wikilinks[token]
                       if o in self.label_encoder.classes_]
            if options:
                # rank scores for these labels and select the highest one as
                # winner:
                scores = prediction[self.label_encoder.transform(options)]
                #logging.debug(
                #    sorted(zip(options, scores), key=itemgetter(1), reverse=True))
                winner, score = sorted(
                    zip(options, scores), key=itemgetter(1), reverse=True)[0]
                if score > 0.80:
                    winner = winner.replace(' ', '_')
                    winners.append(winner)
                else:
                    winners.append('X')    
                #print(token, '>', winner, '>', sorted(zip(options, scores), key=itemgetter(1), reverse=True)[0][1])
            else:
                winners.append('X')
        return winners

    def vectorize_wiki_mentions(self, mentions=(), fresh=False, filename='vectorization.p',
                                ngram_range=(4, 4), max_features=3000):
        """
        Takes mentions, which is a tuple of equal-sized tuples, containing for each mention:
            target_ids, name_variants, left_contexts, right_contexts, page_counts
        Builds/loads a tuple of vectorizers and vectorized data.
        """
        X, y, label_encoder, variant_vectorizer, context_vectorizer, page_cnt_vectorizer = [], [], None, None, None, None

        if fresh:
            if not mentions:
                mentions = self.mentions
            logging.info('Vectorizing %d mentions' % len(mentions[0]))
            # unpack incoming mention data:
            target_ids, name_variants, left_contexts, right_contexts, page_counts = mentions
            # vectorize labels:
            label_encoder = LabelEncoder()
            target_ids = [t.replace('_', ' ') for t in target_ids]
            y = label_encoder.fit_transform(target_ids)
            # vectorize name variants:
            variant_vectorizer = TfidfVectorizer(
                analyzer='char', ngram_range=ngram_range, lowercase=False, max_features=max_features)
            variant_vecs = variant_vectorizer.fit_transform(name_variants)
            # vectorize (left and right) context;
            context_vectorizer = TfidfVectorizer(
                analyzer='char', ngram_range=ngram_range, lowercase=False, max_features=max_features)
            context_vectorizer.fit(left_contexts+right_contexts)
            left_context_vecs = context_vectorizer.transform(left_contexts)
            right_context_vecs = context_vectorizer.transform(right_contexts)
            # vectorize page counts:
            page_cnt_vectorizer = DictVectorizer()
            cnt_vecs = page_cnt_vectorizer.fit_transform(page_counts)
            # concatenate sparse matrices for all feature types:
            X = hstack((variant_vecs, left_context_vecs, right_context_vecs, cnt_vecs))
            #X = variant_vecs
            # dump a tuple of all components
            with open(os.path.join(self.workspace, filename), 'wb') as out:
                pickle.dump((X, y, label_encoder, variant_vectorizer, context_vectorizer,
                             page_cnt_vectorizer), out)

        else:
            logging.info('Loading vectorized mentions...')
            with open(os.path.join(self.workspace, filename), 'rb') as inf:
                X, y, label_encoder, variant_vectorizer, context_vectorizer, page_cnt_vectorizer = pickle.load(inf)

        self.X, self.y, self.label_encoder, self.variant_vectorizer, self.context_vectorizer, self.page_cnt_vectorizer = \
            X, y, label_encoder, variant_vectorizer, context_vectorizer, page_cnt_vectorizer

        logging.info('Vectorized data: %d instances; %d features; %d class labels' % (
            X.shape[0], X.shape[1], len(label_encoder.classes_)))

        # collect variables needed to build the model:
        input_dim = X.shape[1]  # nb features
        output_dim = len(label_encoder.classes_)  # nb labels

        labels = [self.label_encoder.inverse_transform(label) for label in self.y]
        label_cnt = Counter(labels)
        #for l in label_cnt.most_common(500):
        #    print(l)

        return input_dim, output_dim

    def classifier(self, input_dim, output_dim, hidden_dim=1024, fresh=False,
                   filename='classifier.p', test=True, nb_epochs=100):
        """
        Trains a neural net on the vectorized data.
        """

        # define and compile model
        model = neural_model(
            input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
        random_state = 1985

        if fresh:
            X, y = self.X, self.y
            X = X.toarray()  # unsparsify for keras

            logging.info('Applying classifier to %d instances; %d features; %d class labels.' % (
                X.shape[0], X.shape[1], output_dim))

            # convert labels to correct format for cross-entropy loss in keras:
            y = np_utils.to_categorical(y)

            if test:
                # train and test split:
                train_X, test_X, train_y, test_y = train_test_split(
                    X, y, test_size=0.10, random_state=random_state)
                # train and dev split:
                train_X, dev_X, train_y, dev_y = train_test_split(
                    train_X, train_y, test_size=0.10, random_state=random_state)
                # fit and validate:
                model.fit(train_X, train_y, show_accuracy=True, batch_size=100,
                          nb_epoch=nb_epochs, validation_data=(dev_X, dev_y), shuffle=True)
                # test on dev:
                dev_loss, dev_acc = model.evaluate(
                    dev_X, dev_y, batch_size=100, show_accuracy=True, verbose=1)
                logging.info('>>> Dev evaluation:')
                logging.info('\t+ Dev loss:', dev_loss)
                logging.info('\t+ Dev accu:', dev_acc)
                # test on test:
                test_loss, test_acc = model.evaluate(
                    test_X, test_y, batch_size=100, show_accuracy=True, verbose=1)
                logging.info('>>> Test evaluation:')
                logging.info('\t+ Test loss:', test_loss)
                logging.info('\t+ Test accu:', test_acc)
            else:
                # simply train on all data:
                model.fit(X, y, show_accuracy=True, batch_size=100,
                          nb_epoch=nb_epochs, validation_data=None, shuffle=True)

            model.save_weights(
                os.path.join(self.workspace, filename), overwrite=True)

        else:
            model.load_weights(os.path.join(self.workspace, filename))

        self.model = model

        if test:
            return dev_acc, test_acc

    def extract_unique_nes(self, input_dir='frog_periodicals', fresh=False,
                           max_documents=None, max_words_per_doc=None,
                           filename='nes2wikilinks.p', testfiles=[]):
        """
        Extracts all unique entities in the frogged files under input_dir as a dict.
        Registers in this dict: which relevant wiki-pages the NE could refer to
        according to the Wikipedia search interface.
        Only considers NEs that are:
            * capitalized
            * have len > 3 (cf. 'Van')
            * don't end in a full stop (e.g. 'A.F.Th.')
            * tagged as B-PER by Frog
        """
        if fresh:
            logging.info('Extracting NEs from documents!')
            wikipedia = Wikipedia(language='nl', throttle=3)
            self.nes2wikilinks = {}

            for filepath in glob.glob(os.sep.join((self.workspace, input_dir))+'/*.txt.out'):
                if testfiles:
                    fp = os.path.basename(filepath).replace('.txt.out', '')
                    if fp not in testfiles:
                        continue
                max_words = max_words_per_doc
                for line in codecs.open(os.sep.join((self.workspace, filepath)), 'r', 'utf8'):
                    try:
                        comps = [c for c in line.strip().split('\t') if c]
                        idx, token, lemma, pos, conf, ne = comps
                        token = token.replace('_', ' ')
                        if ne.startswith('B-PER') and token[0].isupper() and len(token) > 3 and not token.endswith('.'):
                            if token not in self.nes2wikilinks:
                                try:  # to look up the page in wikipedia:
                                    article = wikipedia.search(token)
                                    if article:  # if we find something...
                                        # we are dealing a referral page
                                        if article.categories[0] == 'Wikipedia:Doorverwijspagina':
                                            for link in article.links:
                                                if link in self.page_ids:
                                                    if token not in self.nes2wikilinks:
                                                        self.nes2wikilinks[token] = set()
                                                    self.nes2wikilinks[token].add(link)
                                        else:
                                            if article.title in self.page_ids:
                                                self.nes2wikilinks[token] = set([article.title])
                                except:  # probably a download issue...
                                    continue
                        max_words -= 1
                        if max_words < 0:
                            break
                    # probably parsing error in the frog file
                    except ValueError:
                        continue

                # update stats:
                max_documents -= 1
                if max_documents % 10 == 0:
                    logging.info('\t+ %d documents to go' % max_documents)
                    logging.info('\t+ %d NEs collected' % len(self.nes2wikilinks))
                if max_documents < 0:
                    break

            with open(os.path.join(self.workspace, filename), 'wb') as out:
                pickle.dump(self.nes2wikilinks, out)

        else:
            with open(os.path.join(self.workspace, filename), 'rb') as inf:
                self.nes2wikilinks = pickle.load(inf)

    def disambiguate_nes(self, max_documents=1000, max_words_per_doc=1000, context_window_size=150,
                         input_dir='frog_periodicals', output_dir='wikified_periodicals', testfiles=[]):
        """
        Loops over frogged files under input_dir.
        First extracts all non-ambiguous NEs, then attempts to disambiguate ambiguous NEs,
        on the basis of token, left and right context + the unambiguous NEs in the doc.
        """
        logging.info(
            'Disambiguating named entities from %d documents!' % max_documents)

        # make sure that we have a fresh output dir:
        if os.path.isdir(os.path.join(self.workspace, output_dir)):
            shutil.rmtree(os.path.join(self.workspace, output_dir))
        os.mkdir(os.path.join(self.workspace, output_dir))

        for filepath in glob.glob(os.path.join(self.workspace, input_dir) + '/*.txt.out'):
            if testfiles:
                fp = os.path.basename(filepath).replace('.txt.out', '')
                if fp not in testfiles:
                    continue
            unambiguous_nes = Counter()  # collect counts of unambiguous NEs
            formatted = ''  # collect tokens in a tmp html format

            for line in codecs.open(filepath, 'r', 'utf8'):
                try:
                    comps = [c for c in line.strip().split('\t') if c]
                    idx, token, lemma, pos, conf, ne = comps
                    if ne.startswith('B-PER') and token[0].isupper() and len(token) > 3 and not token.endswith('.'):
                        token = token.replace('_', ' ')
                        if token in self.nes2wikilinks and len(self.nes2wikilinks[token]) == 1:
                            # only one option, so unambiguous:
                            unambiguous_ne = tuple(
                                self.nes2wikilinks[token])[0]
                            unambiguous_nes[unambiguous_ne] += 1
                            formatted += '<author type="unambiguous" id="' + \
                                unambiguous_ne.replace(' ', '_') + '">' + token + '</author>'

                        elif token in self.nes2wikilinks:
                            formatted += '<author type="ambiguous">' + token + '</author>'

                        else:
                            formatted += token + ' '
                    else:
                        formatted += token + ' '

                except ValueError:
                    continue

            unambiguous_nes = {k: (v/sum(unambiguous_nes.values(), 0.0))
                               for k, v in unambiguous_nes.items()}  # normalize absolute counts

            mentions = self.featurize_section(
                formatted=formatted, page_cnt=unambiguous_nes, context_window_size=context_window_size)

            if mentions:
                X = self.vectorize_dbnl_nes(mentions)
                loc_target_labels, loc_name_variants, loc_left_contexts, loc_right_contexts, loc_page_counts = mentions
                disambiguations = self.classify_nes(X, loc_name_variants)
                #logging.info('%d disambiguations found' % len(disambiguations))
                #for token, disambiguation in zip(loc_name_variants, disambiguations):
                #    logging.debug(token+'> '+disambiguation)

            # second pass over the file; fill in slots in new file:
            new_filename = os.path.join(self.workspace, output_dir, os.path.basename(
                filepath).replace('.txt.out', '.wikified'))
            with codecs.open(new_filename, 'w', 'utf8') as wikified_file:
                for line in codecs.open(filepath, 'r', 'utf8'):
                    try:
                        comps = [c for c in line.strip().split('\t') if c]
                        idx, token, lemma, pos, conf, ne = comps
                        if ne.startswith('B-PER') and token[0].isupper() and len(token) > 3 and not token.endswith('.'):
                            token = token.replace('_', ' ')
                            if token in self.nes2wikilinks and len(self.nes2wikilinks[token]) == 1:
                                # only one option, so unambiguous:
                                unambiguous_ne = tuple(self.nes2wikilinks[token])[0]
                                unambiguous_ne = unambiguous_ne.replace(' ', '_')
                                comps = idx, token, lemma, pos, conf, ne, unambiguous_ne

                            elif token in self.nes2wikilinks:
                                comps = idx, token, lemma, pos, conf, ne, 'X'
                                try:
                                    disambiguated_ne = disambiguations.pop(0)
                                    comps = idx, token, lemma, pos, conf, ne, disambiguated_ne
                                except IndexError:
                                    comps = idx, token, lemma, pos, conf, ne, 'X'

                            else:
                                comps = idx, token, lemma, pos, conf, ne, 'X'
                        else:
                            comps = idx, token, lemma, pos, conf, ne, 'X'

                    except ValueError:
                        continue

                    wikified_file.write('\t'.join(comps)+'\n')

            # update stats:
            max_documents -= 1
            if max_documents % 100 == 0:
                logging.info('\t+ %d documents to go.' % max_documents)
            if max_documents <= 0:
                break
        return

    def naive_disambiguate_nes(self, max_documents=1000, max_words_per_doc=1000, context_window_size=150,
                         input_dir='frog_periodicals', output_dir='wikified_periodicals', testfiles=[]):

        logging.info(
            'Disambiguating named entities from %d documents!' % max_documents)

        # make sure that we have a fresh output dir:
        if os.path.isdir(os.path.join(self.workspace, output_dir)):
            shutil.rmtree(os.path.join(self.workspace, output_dir))
        os.mkdir(os.path.join(self.workspace, output_dir))

        for filepath in glob.glob(os.path.join(self.workspace, input_dir) + '/*.txt.out'):
            if testfiles:
                fp = os.path.basename(filepath).replace('.txt.out', '')
                if fp not in testfiles:
                    continue
            # first pass: collect unambiguous mentions:
            unambiguous_nes = set()
            for line in codecs.open(filepath, 'r', 'utf8'):
                try:
                    comps = [c for c in line.strip().split('\t') if c]
                    idx, token, lemma, pos, conf, ne = comps
                    if ne.startswith('B-PER') and token[0].isupper() and len(token) > 3 and not token.endswith('.'):
                        token = token.replace('_', ' ')
                        if token in self.nes2wikilinks:
                            if len(self.nes2wikilinks[token]) == 1:
                                unambiguous_ne = tuple(self.nes2wikilinks[token])[0]
                                unambiguous_nes.add(unambiguous_ne)
                except ValueError:
                    continue

            new_filename = os.path.join(self.workspace, output_dir, os.path.basename(
                filepath).replace('.txt.out', '.wikified'))
            with codecs.open(new_filename, 'w', 'utf8') as wikified_file:
                for line in codecs.open(filepath, 'r', 'utf8'):
                    try:
                        comps = [c for c in line.strip().split('\t') if c]
                        idx, token, lemma, pos, conf, ne = comps
                        if ne.startswith('B-PER') and token[0].isupper() and len(token) > 3 and not token.endswith('.'):
                            token = token.replace('_', ' ')
                            if token in self.nes2wikilinks:
                                if len(self.nes2wikilinks[token]) == 1:
                                    # only one option, so unambiguous:
                                    unambiguous_ne = tuple(self.nes2wikilinks[token])[0]
                                    comps = idx, token, lemma, pos, conf, ne, unambiguous_ne
                                else:
                                    # try to find anchor:
                                    found = False
                                    for option, _ in self.nes2wikilinks[token]:
                                        if option in unambiguous_nes:
                                            comps = idx, token, lemma, pos, conf, ne, option            
                                            found = True
                                            break
                                    if not found:
                                        comps = idx, token, lemma, pos, conf, ne, 'X'
                            else:
                                comps = idx, token, lemma, pos, conf, ne, 'X'    
                        elif ne == 'O':
                            comps = idx, token, lemma, pos, conf, ne, 'X'
                        else:
                            continue
                    except ValueError:
                        continue
                    comps = list(comps)
                    comps[-1] = comps[-1].replace(' ', '_')
                    wikified_file.write('\t'.join(comps)+'\n')
                    # update stats:
            max_documents -= 1
            if max_documents % 100 == 0:
                logging.info('\t+ %d documents to go.' % max_documents)
            if max_documents <= 0:
                break

    def evaluate_sample(self, corr_dir=None, pred_dir=None):
        if not corr_dir:
            corr_dir = self.workspace + '/corr_rnd_sample'
        if not pred_dir:
            pred_dir = self.workspace + '/wikified_periodicals'
        true, pred = [], []
        for filename in os.listdir(corr_dir):
            if not filename.endswith('.wikified'):
                continue
            corr_lines = codecs.open(corr_dir+'/'+filename, 'r', 'utf8').readlines()
            pred_lines = codecs.open(pred_dir+'/'+filename, 'r', 'utf8').readlines()
            for corr_i, pred_i in zip(corr_lines, pred_lines):
                corr_comps = [c for c in corr_i.strip().split('\t') if c]
                pred_comps = [c for c in pred_i.strip().split('\t') if c]
                # check whether properly aligned:
                if corr_comps[1] == pred_comps[1]:
                    true.append(corr_comps[-1])
                    pred.append(pred_comps[-1])
        f1 = f1_score(true, pred, average='macro')
        print('F1:', f1)
        return f1
        



class DbnlWikifier(Wikifier):
    def __init__(self, workspace=None):
        relevant_categories = """Nederlands_schrijver Nederlands_dichter Vlaams_schrijver Vlaams_dichter Middelnederlands_schrijver
                        Vlaams_dichter_(voor_1830) Vlaams_toneelschrijver Nederlands_toneelschrijver Vlaams_kinderboekenschrijver
                        Nederlands_kinderboekenschrijver""".split()
        Wikifier.__init__(self, workspace=workspace, relevant_categories=relevant_categories)
