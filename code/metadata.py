#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import csv
from collections import Counter
from operator import itemgetter
import glob
import codecs
import re
import os
import numpy as np

from bs4 import BeautifulSoup

import bokeh
from bokeh.charts import Bar
from bokeh.io import output_file, show, vplot, save
from bokeh.plotting import figure
from bokeh.models import Axis

def parse_date_str(date_str):
    """
    Parse a date/year string and return it as an int. If unparsable, returns "???".
    Catches most obvious cases:
    - "1934" > 1934
    - "1934-" > 1934
    - "1934-1938" > 1936
    - "1934-1935" > 1934
    - "ca. 1934" > 1934
    - "ca. 1934-1938" > 1936
    """

    try:

        date_str = date_str.strip()

        # replace ca. and other artefacts:
        if date_str.startswith("ca. "):
            date_str = date_str.replace("ca. ", "").strip()
        date_str = date_str.replace("?", "")
        date_str = date_str.replace("X", "0")
        date_str = date_str.replace("...", "")

        if not date_str:
            return "???"

        if date_str.isdigit():
            if len(date_str) == 4:
                date = int(date_str)
                return date
        elif "-" in date_str:
            dates = [int(y) for y in date_str.split("-") if y]
            if len(dates) == 2:
                date = sum(dates)/2.0
                return int(date)
            else:
                return int(dates[0])
        else:
            return "???"

    except:
        return "???"



def metadata_dict():
    """
    We loop over the metadata files which we have and create for each text an 
    entry in a lookup dict:
        key = text_id's
        values = {genre, subgenre, title, year} (where available)
    This dict only considers files for which we have the actual xml.
    """
    
    meta = dict()

    # define a tuple of the main genres for solving the indexes:
    genres = ("proza", "poëzie", "drama", "sec - letterkunde", "sec - taalkunde", 
          "sec - taal/letterkunde", "jeugdliteratuur", "_",  "non-fictie")

    # add genre info for each text (from titelxgenre.txt):
    with open("../metadata_files/titelxgenre.txt", 'rb') as csvfile:
        for row in csv.reader(csvfile, delimiter=';', quotechar='"'):
            _, text_id, genre_idx = row
            # get the int-encoding of the genres (see dbnl docs):
            genre = genres[int(genre_idx)-1]
            if text_id not in meta:
                meta[text_id] = dict()
            meta[text_id]["genre"] = genre

    # parse int-representation of subgenres (from subgenre.txt):
    subgenre_idxs = {}
    with open("../metadata_files/subgenre.txt", 'rb') as csvfile:
        for row in csv.reader(csvfile, delimiter=';', quotechar='"'):
            subgenre_idx, subgenre_id = row
            subgenre_idxs[subgenre_idx] = subgenre_id

    # add subgenre info on each text (from titelxsubgenre.txt):
    with open("../metadata_files/titelxsubgenre.txt", 'rb') as csvfile:
        for row in csv.reader(csvfile, delimiter=';', quotechar='"'):
            _, text_id, subgenre_idx = row
            if text_id not in meta:
                meta[text_id] = dict()
            meta[text_id]["subgenre"] = subgenre_idxs[subgenre_idx]

    # extract the title and, where possible, an unambiguous date (from titel.txt):
    with open("../metadata_files/titel.txt", 'rb') as csvfile:
        for row in csv.reader(csvfile, delimiter=';', quotechar='"'):
            # title:
            text_id, text_title = row[:2]
            if text_id not in meta:
                meta[text_id] = dict()
            meta[text_id]["title"] = text_title

            # date:
            year = row[3].strip()
            if year:
                meta[text_id]["year"] = parse_date_str(year)
            else:
                pass

    return meta


def main():
    meta = metadata_dict()
    print(meta)

if __name__ == "__main__":
    main()

"""
# stuff for plotting
def plot_nb_texts(metadata_dict={}, text_names=None, filename=''):

    print('orig nb items:', len(metadata_dict))

    if text_names:
        # we only do this for the provided files
        metadata_dict = {k:v for k,v in metadata_dict.items() if k in text_names}
    
    print('actual nb items:', len(metadata_dict))
    
    output_file(filename)
    # plot main genres:
    genre_cnt = Counter((metadata_dict[text]["genre"] for text in metadata_dict if "genre" in metadata_dict[text]))
    genre_items = sorted(genre_cnt.items(), key=itemgetter(1), reverse=True)
    genres, genre_cnts = zip(*genre_items)
    genre_cnts = {"genre_cnts":genre_cnts}
    genres = list(genres)
    
    genre_bar = Bar(genre_cnts, genres, title="Aantal teksten per hoofdcategorie")

    # plot subgenres:
    subgenre_cnt = Counter((metadata_dict[text]["subgenre"] for text in metadata_dict if "subgenre" in metadata_dict[text]))
    subgenre_items = sorted(subgenre_cnt.items(), key=itemgetter(1), reverse=True)[:15] # only plot the 10 most common ones
    subgenres, subgenre_cnts = zip(*subgenre_items)
    subgenres = list([unicode(BeautifulSoup(s).text) for s in subgenres])
    subgenre_cnts = {"subgenre_cnts":subgenre_cnts}
    subgenre_bar = Bar(subgenre_cnts, subgenres, title="Aantal teksten per subcategorie")
    xaxis = subgenre_bar.select(dict(type=Axis))[0]
    xaxis.major_label_standoff = 0
    xaxis.major_label_orientation = np.pi/2
    xaxis.major_label_standoff = 6
    xaxis.major_tick_out = 0

    p = vplot(genre_bar, subgenre_bar)
    save(p)

def plot_size_texts(metadata, filename):
    output_file(filename)
    genres = ("proza", "poëzie", "drama", "sec - letterkunde", "sec - taalkunde", 
          "sec - taal/letterkunde", "jeugdliteratuur", "_",  "non-fictie")
    genre_sizes = {k:0 for k in genres}

    subgenre_sizes = {}
    with open("../metadata_files/subgenre.txt", 'rb') as csvfile:
        for row in csv.reader(csvfile, delimiter=';', quotechar='"'):
            _, subgenre_id = row
            subgenre_sizes[subgenre_id] = 0
    
    for filepath in glob.glob('../texts/*.xml')[:10000]:
        filename = os.path.splitext(os.path.basename(filepath))[0]
        text_id = filename[:-3] # remove trailing "_01"
        if text_id in metadata:
            if 'genre' in metadata[text_id]:
                genre_sizes[metadata[text_id]['genre']] += os.path.getsize(filepath)
            if 'subgenre' in metadata[text_id]:
                subgenre_sizes[metadata[text_id]['subgenre']] += os.path.getsize(filepath)

    genre_sizes = {k:(v/float(1000000000)) for k,v in genre_sizes.items() if v} # convert to megabytes
    genre_items = sorted(genre_sizes.items(), key=itemgetter(1), reverse=True)
    genres, genre_cnts = zip(*genre_items)
    genre_cnts = {"genre_cnts":genre_cnts}
    genres = list(genres)
    genre_bar = Bar(genre_cnts, genres)

    subgenre_sizes = {k:(v/float(1000000000)) for k,v in subgenre_sizes.items() if v} # convert to Megabytes
    subgenre_items = sorted(subgenre_sizes.items(), key=itemgetter(1), reverse=True)[:15] # only plot the 10 most common ones
    subgenres, subgenre_cnts = zip(*subgenre_items)
    subgenres = list([unicode(BeautifulSoup(s).text) for s in subgenres])
    subgenre_cnts = {"subgenre_cnts":subgenre_cnts}
    subgenre_bar = Bar(subgenre_cnts, subgenres, title="Hoeveelheid tekst per subcategorie (GB)")
    xaxis = subgenre_bar.select(dict(type=Axis))[1]
    xaxis.major_label_standoff = 0
    xaxis.major_label_orientation = np.pi/2
    xaxis.major_label_standoff = 6
    xaxis.major_tick_out = 0
    p = vplot(genre_bar, subgenre_bar)
    save(p)

def clean_journal_title(title):
    title = title.replace('ZL.', 'Zacht Lawijd.')
    title = title.replace('ZL.', 'Zacht Lawijd.')
    title = title.replace(' (nieuwe reeks)', '')
    date = re.compile(r'\s*[0-9]+\-*\s*')
    title = date.sub('', title)
    if '.' in title:
        title = title.split('.')[0]
    title = title.replace(', ', '')
    return title

def plot_periodicals(metadata, filename='periodicals.html'):
    output_file(filename)
    journal_counts = Counter()
    for filepath in glob.glob('../texts/*.xml')[:10000]:
        filename = os.path.splitext(os.path.basename(filepath))[0]
        text_id = filename[:-3] # remove trailing "_01"
        if text_id in metadata:
            if 'subgenre' in metadata[text_id]:
                if metadata[text_id]['subgenre'] == 'tijdschrift / jaarboek':
                    title = metadata[text_id]['title']
                    journal_counts[clean_journal_title(title)] += 1
    journal_items = sorted(journal_counts.items(), key=itemgetter(1), reverse=True)[:15]
    print(journal_items)
    p = figure(plot_width=400, plot_height=400)
    p.xaxis.major_label_orientation = "vertical"
    journals, journal_cnts = zip(*journal_items)
    clean_names = []
    for title in journals:
        title = unicode(BeautifulSoup(title).text)
        print(title)
        if len(title) > 30:
            title = "".join([w[0] for w in title.split() if w[0].isupper()])
            print(title)
        clean_names.append(title)
    print(clean_names)
    journal_cnts = {"journal_cnts":journal_cnts}
    journal_bar = Bar(journal_cnts, clean_names, title="Cumulatief # tijdschriftnummers")
    show(journal_bar)
    save(journal_bar)

def plot_frogged_data(metadata, filename='frogged.html'):
    output_file(filename)
    texts_per_year = {str(k):0 for k in range(1946, 2011)}
    words_per_year = {str(k):0 for k in range(1946, 2011)}

    iter_cnt = 0
    for filepath in glob.glob('../workspace/frog_periodicals/*.txt.out'):
        filename = os.path.basename(filepath)
        year = filename.replace('.txt.out', '').split('-')[-1]
        cnt = 0
        for line in codecs.open(filepath, 'r', 'utf8'):
            line = line.strip()
            if line:
                cnt += 1
        try:
            words_per_year[year] += cnt
            texts_per_year[year] += 1
        except KeyError:
            pass
        iter_cnt += 1
        if iter_cnt % 1000 == 0:
            print(iter_cnt, 'documents parsed')
        #if iter_cnt >= 2000:
        #    break

    print('total nb of words in corpus', sum(words_per_year.values()))
    print('total nb of articles in corpus', sum(texts_per_year.values()))

    word_counts = {k:v for k,v in words_per_year.items() if v} # convert to megabytes
    word_items = sorted(word_counts.items(), key=itemgetter(0), reverse=False)
    years, word_cnts = zip(*word_items)
    word_cnts = {"word_cnts":word_cnts}
    years = list(years)
    word_bar = Bar(word_cnts, years, title='# woorden per jaar (1945-2010)', width=1400, height=400)
    xaxis = word_bar.select(dict(type=Axis))[1]
    xaxis.major_label_standoff = 0
    xaxis.major_label_orientation = np.pi/2
    xaxis.major_label_standoff = 6
    xaxis.major_tick_out = 0

    text_counts = {k:v for k,v in texts_per_year.items() if v} # convert to megabytes
    text_items = sorted(text_counts.items(), key=itemgetter(0), reverse=False)
    years, text_cnts = zip(*text_items)
    text_cnts = {"text_cnts":text_cnts}
    years = list(years)
    text_bar = Bar(text_cnts, years, title="# 'artikels' per jaar (1945-2010)", width=1400, height=400)
    xaxis = text_bar.select(dict(type=Axis))[1]
    xaxis.major_label_standoff = 0
    xaxis.major_label_orientation = np.pi/2
    xaxis.major_label_standoff = 6
    xaxis.major_tick_out = 0
    p = vplot(word_bar, text_bar)
    save(p)

#actual_texts = []
#for filepath in glob.glob('../texts/*.xml'):
#    text_id = os.path.splitext(os.path.basename(filepath))[0][:-3] # remove trailing "_01"
#    actual_texts.append(text_id)
d = metadata_dict()
print(d)
#plot_nb_texts(d, actual_texts, 'counts.html')
#plot_size_texts(d, 'sizes.html')
#plot_periodicals(d, 'periodicals.html')
#plot_frogged_data(d, 'frogged.html') # must be run on server
"""



    
