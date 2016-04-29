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
    with open("../metadata_files/titelxgenre.txt", 'r', encoding='utf-8') as csvfile:
        for row in csv.reader(csvfile, delimiter=';', quotechar='"'):
            try:
                _, text_id, genre_idx = row
                # get the int-encoding of the genres (see dbnl docs):
                genre = genres[int(genre_idx)-1]
                if text_id not in meta:
                    meta[text_id] = dict()
                meta[text_id]["genre"] = genre
            except UnicodeDecodeError:
                pass

    # parse int-representation of subgenres (from subgenre.txt):
    subgenre_idxs = {}
    with open("../metadata_files/subgenre.txt", 'r', encoding='utf-8') as csvfile:
        for row in csv.reader(csvfile, delimiter=';', quotechar='"'):
            try:
                subgenre_idx, subgenre_id = row
                subgenre_idxs[subgenre_idx] = subgenre_id
            except UnicodeDecodeError:
                pass

    # add subgenre info on each text (from titelxsubgenre.txt):
    with open("../metadata_files/titelxsubgenre.txt", 'r', encoding='utf-8') as csvfile:
        for row in csv.reader(csvfile, delimiter=';', quotechar='"'):
            try:
                _, text_id, subgenre_idx = row
                if text_id not in meta:
                    meta[text_id] = dict()
                meta[text_id]["subgenre"] = subgenre_idxs[subgenre_idx]
            except UnicodeDecodeError:
                pass

    # extract the title and, where possible, an unambiguous date (from titel.txt):
    with open("../metadata_files/titel.txt", 'r', encoding='utf-8') as csvfile:
        for row in csv.reader(csvfile, delimiter=';', quotechar='"'):
            try:
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
            except UnicodeDecodeError:
                pass

    return meta



    
