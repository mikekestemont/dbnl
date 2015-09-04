#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function


import os
import glob
import re
import shutil
import codecs
import subprocess
from operator import itemgetter
from collections import Counter

from lxml import etree
from bs4 import BeautifulSoup
from bokeh.plotting import output_file, figure, save
from langdetect import detect

import metadata


# some regular expressions:
whitespace = re.compile(r'\s+')
nasty_pagebreaks = re.compile(r'\s*-<pb n="[0-9]+"></pb>\s*')


def xml_to_articles(filepath):
    """
    Parses an xml and returns plain text versions of the individual chapters (i.e. articles, reviews, ...)
    in the file. Only files are let through that are
        - recognized as 'nl', with a probability above .30.
        - longer than 200 characters
    """

    articles = []

    xml_str = codecs.open(os.path.abspath(filepath), 'r', 'utf8').read()
    xml_str = unicode(BeautifulSoup(xml_str)) # remove entities from the xml

    # get rid of nasty pagebreak (pb), breaking up tokens across pages:
    xml_str = re.sub(nasty_pagebreaks, '', xml_str)

    # attempt to parse the tree:
    try:
        tree = etree.fromstring(xml_str)
    except etree.XMLSyntaxError:
        return None

    # remove cf- and note-elements (don't contain actual text):
    for element in tree.xpath(".//cf"):
        element.getparent().remove(element)
    
    # individual articles etc. are represented as div's which have the type-attribute set to 'chapter':
    chapter_nodes = [node for node in tree.findall('.//div')
                        if node.attrib and \
                           'type' in node.attrib and \
                           node.attrib['type'] == 'chapter']

    for chapter_node in chapter_nodes:
        # all text in the articles is contained under p-elements:
        article_text = ""
        for p_node in chapter_node.findall('.//p'):
            # remove elements that contain meta text (note that we exclude all notes!)
            for tag_name in ('note', 'figure', 'table'):
                etree.strip_elements(p_node, tag_name, with_tail=False)

            # collect the actual text:
            p_text = "".join(p_node.itertext())

            # add the article (and add some whitespace to be safe):
            article_text += p_text+" "

        # collapse all whitespace to single spaces:
        article_text = re.sub(whitespace, ' ', article_text).strip()

        if len(article_text) > 500:
            if detect(article_text) == 'nl':
                articles.append(article_text)
            #else:
            #    print(article_text[:200])
            #    print(detect(article_text))

    return articles


def parse_secondary_dbnl(max_documents=100):
    """
    Parses all xml-files under the ../texts directory.
    Only considers files with:
        - genre = 'sec - letterkunde'
        - subgenre = 'tijdschrift / jaarboek'
        - 1945 > date < 2002
    Additionally, only Dutch-language articles will be included.
    Only outputs articles which are recognized as 'nl'
    All individual 'chapters' (i.e. articles) are saved separately in ../workspace/periodicals
    """

    year_counts = Counter()

    # get metadata
    metadata_dict = metadata.metadata_dict()
    
    # keep track:
    document_cnt = 0 # nb of documents (i.e. 'journal issues')
    article_cnt = 0 # nb of chapters (i.e. 'articles/reviews')

    # initalize directories:
    if not os.path.isdir('../workspace'):
        os.mkdir('../workspace')
    if not os.path.isdir('../figures'):
        os.mkdir('../figures')
    if os.path.isdir('../workspace/periodicals'):
        shutil.rmtree('../workspace/periodicals')
    os.mkdir('../workspace/periodicals')

    # iterate over the full texts which we have:
    for filepath in glob.glob('../texts/*.xml'):

        text_id = os.path.splitext(os.path.basename(filepath))[0][:-3] # remove trailing "_01"

        # see whether we have all the necessary metadata for the text:
        try:
            title = metadata_dict[text_id]['title']
            date = metadata_dict[text_id]['year']
            genre = metadata_dict[text_id]['genre']
            subgenre = metadata_dict[text_id]['subgenre']
        except KeyError:
            continue

        # limited to post-war studies on literature in periodicals:
        if genre == 'sec - letterkunde' and \
            subgenre == 'tijdschrift / jaarboek' and \
            date > 1945 and date < 2002 and date != "???":

            print(">>>", title)

            # collect the individual articles in the issue:
            articles = xml_to_articles(filepath)
            if articles:
                for idx, article in enumerate(articles):
                    new_filepath = '../workspace/periodicals/'
                    new_filepath += text_id+"-"+str(idx+1)+'-'+str(date)+'.txt'
                    with codecs.open(new_filepath, 'w', 'utf-8') as f:
                        f.write(article)

                    # update stats:
                    article_cnt += 1
                    year_counts[date] += 1

            # update cnts:
            document_cnt += 1

        if document_cnt >= max_documents:
            break

    print('nb issues parsed:', document_cnt)
    print('nb individual articles extracted:', article_cnt)

    # visualize distribution over time:
    cnts = sorted(year_counts.items())
    output_file('../figures/nb_articles_yearly.html')
    p = figure(plot_width=1200, plot_height=400, x_axis_label='year', y_axis_label='nb articles')
    p.line([y for y,_ in cnts], [c for _,c in cnts], line_width=2)
    save(p)


def frog_articles():
    """
    Use frog to tag etc. the plain text articles under ../workspace/periodicals
    and save the output to ../workspace/frog_periodics.
    Internal note: has to be run on serv1
    """

    # create/flush output dir if necessary:
    if os.path.isdir('../workspace/frog_periodicals'):
        shutil.rmtree('../workspace/frog_periodicals')
    os.mkdir('../workspace/frog_periodicals')

    output_dir = os.path.abspath('../workspace/frog_periodicals')
    input_dir = os.path.abspath('../workspace/periodicals')

    # create cmd line str for frog; don't do morphological analyses etc.
    frog_cmd_str = "/usr/bin/frog --testdir="+input_dir+" --outputdir="+output_dir+" --skip=acp"
    subprocess.call(frog_cmd_str, shell=True)








