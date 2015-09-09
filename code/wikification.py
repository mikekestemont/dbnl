#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import cPickle as pickle
import re
import codecs
import glob
import os
import shutil
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

from utils import query, _plaintext, neural_model


class Wikifier(object):

    def __init__(self, relevant_categories=None):
        """
        Intialize wikifier with the relevant categories you wish to extract.
        """
        if relevant_categories == None:
            self.relevant_categories = """Nederlands_schrijver Nederlands_dichter Vlaams_schrijver Vlaams_dichter Middelnederlands_schrijver
                        Vlaams_dichter_(voor_1830) Vlaams_toneelschrijver Nederlands_toneelschrijver Vlaams_kinderboekenschrijver
                        Nederlands_kinderboekenschrijver""".split()
        else:
            self.relevant_categories = relevant_categories
        # make sure that we have a workspace where we can pickle:
        if not os.path.isdir('../workspace/'):
            os.mkdir('../workspace')


    def relevant_page_ids(self, fresh=False, filename='../workspace/relevant_page_ids.p'):
        """
        Collect the wiki id's (page_ids) for all pages belonging to the self.relevant_categories
            * if fresh: fresh download from Wikipedia + pickled under filename
            * else: no download; page_ids loaded from pickle in filename
        """
        print('>>> Getting page_ids from relevant categories')
        self.page_ids = set()

        if fresh:
            for category in self.relevant_categories:
                for result in query( {'generator':'categorymembers', 'gcmtitle':'Category:'+category, 'gcmlimit':'500'}):
                    for page in result['pages']:
                        page_id = result['pages'][page]['title']
                        self.page_ids.add(page_id)
                        if len(self.page_ids)%1000 == 0:
                            print('\t+ nb pages download:', len(self.page_ids))

            self.page_ids = sorted(self.page_ids)
            pickle.dump(self.page_ids, open(filename, 'wb'))

        else:
            self.page_ids = pickle.load(open(filename, 'rb'))

        print('\t+ set', len(self.page_ids), 'page_ids')

        return self.page_ids


    def backlinking_pages(self, page_ids=None, ignore_categories=None, fresh=False, filename='../workspace/backlinks.p'):
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
            print('>>> Collecting backlinks for', len(page_ids), 'pages')

            if not ignore_categories:
                ignore_categories = ('Gebruiker:', 'Lijst van', 'Portaal:', 'Overleg', 'Wikipedia:', 'Help:', 'Categorie:')

            for idx, page_id in enumerate(page_ids):
                self.backlinks[page_id] = set()
                for result in query({'action':'query', 'list':'backlinks', 'format':'json', 'bltitle':page_id}):
                    for backlink in result['backlinks']:
                        backlink = backlink['title'].replace('_', ' ') # clean up
                        if not backlink.startswith(ignore_categories):
                            self.backlinks[page_id].add(backlink)
                if idx % 10 == 0:
                    print('\t+ collected', sum([len(v) for k,v in self.backlinks.items()]), 'backlinks for', idx+1, 'pages')

            self.backlinks = {k:v for k,v in self.backlinks.items() if v} # remove pages without relevant backlinks
            pickle.dump(self.backlinks, open(filename, 'wb')) # dump for later reuse

        else:
            self.backlinks = pickle.load(open(filename, 'rb'))

        print('\t+ loaded', sum([len(v) for k,v in self.backlinks.items()]), 'backlinks for', len(self.backlinks), 'pages')


    def mentions_from_backlinks(self, backlinks={}, fresh=False, filename='../workspace/mentions.p', context_window_size=150):
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
            print('>>> Mining mentions from', sum([len(v) for k,v in backlinks.items()]),
                  'backlinking pages to', len(backlinks), 'target pages')
            print(backlinks)
            wikipedia = Wikipedia(language='nl', throttle=2)

            for idx, (page_id, links) in enumerate(backlinks.items()):
                print('\t + mining mentions of', page_id, '('+str(len(links)), 'backlinks) | page', idx+1, '/', len(backlinks))
                for backlink in links:
                    try:
                        article = wikipedia.search(backlink) # fetch the linking page via pattern
                        if not article.categories[0] == 'Wikipedia:Doorverwijspagina': # skip referral pages
                            print('\t\t* backlink:', backlink)
                            section_sources = [] # fetch the html-sections of individual sections:
                            if not article.sections: # article doesn't have sections
                                section_sources = [article.source]
                            else:
                                section_sources = [section.source for section in article.sections]
                            # loop over the section sources and extract all relevant mentions:
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
                    except:
                        continue

            pickle.dump((target_ids, name_variants, left_contexts, right_contexts, page_counts), open(filename, 'wb'))

        else:
            target_ids, name_variants, left_contexts, right_contexts, page_counts = \
                                                        pickle.load(open(filename, 'rb'))

        self.mentions = (target_ids, name_variants, left_contexts, right_contexts, page_counts)


    def featurize_section(self, formatted, page_cnt, context_window_size):
        """
        Takes a string in which author NEs have been tagged as <author>variant</author>
        Returns tuples with for each author: target_id, name_variant, left_context, right_context, page_count
        Ignores author-elements that have an attribute type=unambiguous
        """
        loc_target_labels, loc_name_variants, loc_left_contexts, loc_right_contexts, loc_page_counts = [], [], [], [], []

        formatted = formatted.replace('-', ' ') # replace hyphens for date series such as "1940-1945" > "1940 1945"
        # keep minimum of chars (to avoid encoding errors)
        formatted = "".join([char for char in formatted if \
                        (char.isalpha() or char.isdigit() or char.isspace() or char in '<>/=_.-"' or char == "'")])
        # add root tags for parsability:
        formatted = '<root>'+formatted+'</root>'

         # parse the xml representation of the section:
        try:
            tree = minidom.parseString(formatted.encode('utf-8'))
        except:
            return None

        for node in tree.getElementsByTagName('author'):
            if 'type' in node.attributes.keys() and node.attributes['type'].value == 'unambiguous':
                # skip these during testing, where we don't disambiguate all NEs:
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
                        left_context = prev_node.data + left_context # mind the order!
                    else:
                        left_context = prev_node.firstSibling.data + left_context # mind the order!
                except AttributeError: # end of tree (no next)
                    break
            left_context = ' '.join(left_context.strip().split())[-context_window_size:] # cut
            
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
                except AttributeError: # beginning of tree (no next)
                    break
            right_context = ' '.join(right_context.strip().split())[:context_window_size] # cut
            
            try:
                # append to containers:
                loc_name_variants.append("%"+node.firstChild.data+"$") # extract variant name; mark beginning and ending of strings
                if 'id' in node.attributes.keys():
                    loc_target_labels.append(node.attributes['id'].value) # extract class label, if available
                else:
                    loc_target_labels.append('<unk>')
                loc_left_contexts.append(left_context)
                loc_right_contexts.append(right_context)
                loc_page_counts.append(page_cnt) # note that this cntr is the same for all mentions in the same section
            except AttributeError:
                pass
        if loc_name_variants:
            return loc_target_labels, loc_name_variants, loc_left_contexts, loc_right_contexts, loc_page_counts
        else:
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
        
        # count other mentions of pages in self.page_ids in this section (for global disambiguation):
        page_cnt = Counter()
        soup = BeautifulSoup('<html>'+source+'</html>', 'html.parser') # add tags for parsability
        for a_node in soup.find_all('a'): # iterate over all hyperlinks
            try:
                link, title = a_node.get('href'), a_node.get('title').replace('_', ' ')
                if link.startswith('/wiki/') and title in self.page_ids: # check whether the link is of interest
                    page_cnt[title] += 1
            except AttributeError:
                pass
        page_cnt = {k:(v/sum(page_cnt.values(), 0.0)) for k,v in page_cnt.items()} # normalize absolute counts

        # convert html to plain text, but preserve links (a-elements):
        clean_text_with_links = plaintext(html=source, keep={'a':['title', 'href']})

        # reformat relevant author links to tmp syntax (%Hugo_Claus|Claus%):
        author_pattern = re.compile(r'<a href="/wiki/([^\"]*)" title="'+target_id+'">([^\<]*)</a>')
        formatted = author_pattern.sub(repl=r'%\1|\2%', string=clean_text_with_links)
        # now remove remaining, irrelevant links (e.g. external links):
        formatted = _plaintext(formatted)
        # and reformat into xml:
        author_pattern = re.compile(r'\%([^\%]{0,30})\|([^\%]{0,30})\%') # correct for irregularities (too long a name variant)
        formatted = author_pattern.sub(repl=r'<author id="\1">\2</author>', string=formatted)

        featurized = self.featurize_section(formatted, page_cnt, context_window_size)
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
        return X # still sparse!


    def classify_nes(self, X, original_tokens):
        """
        Takes the vectorized matrix X for a list of new tokens and returns the disambiguated pages
        """
        winners = []
        predictions = self.model.predict(X.toarray()) # unsparsify
        original_tokens = [t[1:-1] for t in original_tokens] # rm special symbols surrounding strings
        for prediction, token in zip(predictions, original_tokens):
            # only consider class labels returned by the wikipedia search for token:
            options = [o for o in self.nes2wikilinks[token] if o in self.label_encoder.classes_]
            if options:
                # rank scores for these labels and select the highest one as winner:
                scores = prediction[self.label_encoder.transform(options)]
                #print(sorted(zip(options, scores), key=itemgetter(1), reverse=True))
                winner = sorted(zip(options, scores), key=itemgetter(1), reverse=True)[0][0]
                winner = winner.replace(' ', '_')
                winners.append(winner)
            else:
                winners.append('<unk>')
        return winners


    def vectorize_wiki_mentions(self, mentions=(), fresh=False, filename='../workspace/vectorization.p',
                                ngram_range=(4,4), max_features=3000):
        """
        Takes mentions, which is a tuple of equal-sized tuples, containing for each mention:
            target_ids, name_variants, left_contexts, right_contexts, page_counts
        Builds/loads a tuple of vectorizers and vectorized data.
        """
        X, y, label_encoder, variant_vectorizer, context_vectorizer, page_cnt_vectorizer = [], [], None, None, None, None

        if fresh:
            if not mentions:
                mentions = self.mentions
            print('>>> Vectorizing', len(mentions[0]), 'mentions')
            # unpack incoming mention data:
            target_ids, name_variants, left_contexts, right_contexts, page_counts = mentions
            # vectorize labels:
            label_encoder = LabelEncoder()
            target_ids = [t.replace('_', ' ') for t in target_ids]
            y = label_encoder.fit_transform(target_ids)
            # vectorize name variants:
            variant_vectorizer = TfidfVectorizer(analyzer='char', ngram_range=ngram_range, lowercase=False, max_features=max_features)
            variant_vecs = variant_vectorizer.fit_transform(name_variants)
            # vectorize (left and right) context;
            context_vectorizer = TfidfVectorizer(analyzer='char', ngram_range=ngram_range, lowercase=False, max_features=max_features)
            context_vectorizer.fit(left_contexts+right_contexts)
            left_context_vecs = context_vectorizer.transform(left_contexts)
            right_context_vecs = context_vectorizer.transform(right_contexts)
            # vectorize page counts:
            page_cnt_vectorizer = DictVectorizer()
            cnt_vecs = page_cnt_vectorizer.fit_transform(page_counts)
            # concatenate sparse matrices for all feature types:
            X = hstack((variant_vecs, left_context_vecs, right_context_vecs, cnt_vecs))
            # dump a tuple of all components
            pickle.dump((X, y, label_encoder, variant_vectorizer, context_vectorizer, page_cnt_vectorizer), open(filename, 'wb'))

        else:
            print('\t+ Loading vectorized mentions...')
            X, y, label_encoder, variant_vectorizer, context_vectorizer, page_cnt_vectorizer = pickle.load(open(filename, 'rb'))

        self.X, self.y, self.label_encoder, self.variant_vectorizer, self.context_vectorizer, self.page_cnt_vectorizer = \
            X, y, label_encoder, variant_vectorizer, context_vectorizer, page_cnt_vectorizer           

        print('>>> Vectorized data:', X.shape[0], 'instances;', X.shape[1], 'features;', len(label_encoder.classes_), 'class labels') 

        # collect variables needed to build the model:
        input_dim = X.shape[1] # nb features
        output_dim = len(label_encoder.classes_) # nb labels

        return input_dim, output_dim


    def classifier(self, input_dim, output_dim, hidden_dim=1024, fresh=False,
                   filename='../workspace/classifier.p', test=True, nb_epochs=100):
        """
        Trains a neural net on the vectorized data.
        """

        # define and compile model
        model = neural_model(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
        random_state = 1985

        if fresh:
            X, y = self.X, self.y
            X = X.toarray() # unsparsify for keras
            print('>>> Applying classifier to', X.shape[0], 'instances;', X.shape[1], 'features;', output_dim, 'class labels') 
            
            # convert labels to correct format for cross-entropy loss in keras:
            y = np_utils.to_categorical(y)

            if test:
                # train and test split:
                train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.10, random_state=random_state)
                # train and dev split:
                train_X, dev_X, train_y, dev_y = train_test_split(train_X, train_y, test_size=0.10, random_state=random_state)
                # fit and validate:
                model.fit(train_X, train_y, show_accuracy=True, batch_size=100, nb_epoch=nb_epochs, validation_data=(dev_X, dev_y), shuffle=True)
                # test on dev:
                dev_loss, dev_acc = model.evaluate(dev_X, dev_y, batch_size=100, show_accuracy=True, verbose=1)
                print('>>> Dev evaluation:')
                print('\t+ Test loss:', dev_loss)
                print('\t+ Test accu:', dev_acc)
                # test on test:
                test_loss, test_acc = model.evaluate(test_X, test_y, batch_size=100, show_accuracy=True, verbose=1)
                print('>>> Test evaluation:')
                print('\t+ Test loss:', test_loss)
                print('\t+ Test accu:', test_acc)
            else:
                # simply train on all data:
                model.fit(X, y, show_accuracy=True, batch_size=100, nb_epoch=nb_epochs, validation_data=None, shuffle=True)

            model.save_weights(filename, overwrite=True)

        else:
            model.load_weights(filename)

        self.model = model

        if test:
            return dev_acc, test_acc


    def extract_unique_nes(self, input_dir='../workspace/frog_periodicals', fresh=False,
                            max_documents=None, max_words_per_doc=None,
                            filename='../workspace/nes2wikilinks.p'):
        """
        Extracts all unique entities in the frogged files under input_dir as a dict.
        Registers in this dict: which relevant wiki-pages the NE could refer to
        according to the Wikipedia search interface.
        Only considers NEs that are:
            * capitalized
            * have len > 3 (cf. 'Van')
            * don't end in a dot (e.g. 'A.F.Th.')
            * tagged as B-PER by Frog
        """
        if fresh:
            print('Extracting NEs from ', max_documents, 'documents!')
            wikipedia = Wikipedia(language='nl', throttle=3)
            self.nes2wikilinks = {}

            for filepath in glob.glob(input_dir+'/*.txt.out'):
                max_words = max_words_per_doc
                for line in codecs.open(filepath, 'r', 'utf8'):
                    try:
                        comps = [c for c in line.strip().split('\t') if c]
                        idx, token, lemma, pos, conf, ne  = comps
                        token = token.replace('_', ' ')
                        if ne.startswith('B-PER') and token[0].isupper() and len(token) > 3 and not token.endswith('.'):
                            if token not in self.nes2wikilinks:
                                try: # to look up the page in wikipedia:
                                    article = wikipedia.search(token)
                                    if article: # if we find something...
                                        if article.categories[0] == 'Wikipedia:Doorverwijspagina': # we are dealing a referral page
                                            for link in article.links:
                                                if link in self.page_ids:
                                                    if token not in self.nes2wikilinks:
                                                        self.nes2wikilinks[token] = set()
                                                    self.nes2wikilinks[token].add(link)
                                        else:
                                            if article.title in self.page_ids:
                                                self.nes2wikilinks[token] = set([article.title])
                                except: # probably a download issue...
                                    continue
                        max_words -= 1
                        if max_words < 0:
                            break
                    except ValueError: # probably parsing error in the frog file
                        continue

                # update stats:
                max_documents -= 1
                if max_documents % 10 == 0:
                    print('\t+ ', max_documents, 'documents to go')
                    print('\t+ ', len(self.nes2wikilinks), 'NEs collected')
                if max_documents < 0:
                    break

            pickle.dump(self.nes2wikilinks, open(filename, 'wb'))

        else:
            self.nes2wikilinks = pickle.load(open(filename, 'rb'))


    def disambiguate_nes(self, max_documents=1000, max_words_per_doc=1000, context_window_size=150,
                         input_dir='../workspace/frog_periodicals', output_dir='../workspace/wikified_periodicals'):
        """
        Loops over frogged files under input_dir.
        First extracts all non-ambiguous NEs, then attempts to disambiguate ambiguous NEs,
        on the basis of token, left and right context + the unambiguous NEs in the doc.
        """
        print('Disambiguating named entities from', max_documents, 'documents!')

        # make sure that we have a fresh output dir:
        if os.path.isdir(output_dir):
            shutil.rmtree(output_dir)
        os.mkdir(output_dir)

        for filepath in glob.glob(input_dir+'/*.txt.out'):
            unambiguous_nes = Counter() # collect counts of unambiguous NEs
            formatted = '' # collect tokens in a tmp html format

            for line in codecs.open(filepath, 'r', 'utf8'):
                try:
                    comps = [c for c in line.strip().split('\t') if c]
                    idx, token, lemma, pos, conf, ne  = comps
                    if ne.startswith('B-PER') and token[0].isupper() and len(token) > 3 and not token.endswith('.'):
                        token = token.replace('_', ' ')
                        if token in self.nes2wikilinks and len(self.nes2wikilinks[token]) == 1:
                            # only one option, so unambiguous:
                            unambiguous_ne = tuple(self.nes2wikilinks[token])[0]
                            unambiguous_nes[unambiguous_ne] += 1
                            formatted += '<author type="unambiguous" id="'+unambiguous_ne.replace(' ', '_')+'">'+token+'</author>'
                            
                        elif token in self.nes2wikilinks:
                            formatted += '<author type="ambiguous">'+token+'</author>'

                        else:
                            formatted += token+' '    
                    else:
                        formatted += token+' '

                except ValueError:
                    continue

            unambiguous_nes = {k:(v/sum(unambiguous_nes.values(), 0.0)) for k,v in unambiguous_nes.items()} # normalize absolute counts
            
            mentions = self.featurize_section(formatted=formatted, page_cnt=unambiguous_nes, context_window_size=context_window_size)

            if mentions:
                X = self.vectorize_dbnl_nes(mentions)
                loc_target_labels, loc_name_variants, loc_left_contexts, loc_right_contexts, loc_page_counts = mentions
                disambiguations = self.classify_nes(X, loc_name_variants)
                print(len(disambiguations), 'disambiguations found')
                for token, disambiguation in zip(loc_name_variants, disambiguations):
                    print(token, '>', disambiguation)

            # second pass over the file; fill in slots in new file:
            new_filename = output_dir + '/' + os.path.basename(filepath).replace('.txt.out', '.wikified')
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
                                comps = (idx, token, lemma, pos, conf, ne, unambiguous_ne)
                                
                            elif token in self.nes2wikilinks:
                                disambiguated_ne = disambiguations.pop(0)
                                comps = (idx, token, lemma, pos, conf, ne, disambiguated_ne)

                            else:
                                comps = (idx, token, lemma, pos, conf, ne, 'X')
                        else:
                            comps = (idx, token, lemma, pos, conf, ne, 'X')

                    except ValueError:
                        continue

                    wikified_file.write('\t'.join(comps)+'\n')

            # update stats:
            max_documents -= 1
            if max_documents % 100 == 0:
                print('\t+', max_documents, 'documents to go')
            if max_documents <= 0:
                break

            # TO DO: this function should append the wikifier's output to the original columns in the frog data for easy reuse.
        return




