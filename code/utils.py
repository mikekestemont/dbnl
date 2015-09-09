#!/usr/bin/env python
# -*- coding: utf-8 -*-

import requests
import re

from pattern.web import strip_element, plaintext

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout

def query(request):
    request['action'] = 'query'
    request['format'] = 'json'
    lastContinue = {'continue': ''}

    while True:
        # Clone original request
        req = request.copy()
        # Modify it with the values returned in the 'continue' section of the last result.
        req.update(lastContinue)
        # Call API
        result = requests.get('http://nl.wikipedia.org/w/api.php', params=req).json()
        if 'error' in result: raise ValueError(result['error'])
        if 'warnings' in result: print(result['warnings'])
        if 'query' in result: yield result['query']
        if 'continue' not in result: break
        lastContinue = result['continue']

def _plaintext(string):
    """
    Stole this function in slightmy modified form from pattern where it lives as a class method:
    
    """
    """ Strips HTML tags, whitespace and wiki markup from the HTML source, including:
        metadata, info box, table of contents, annotations, thumbnails, disambiguation link.
        This is called internally from MediaWikiArticle.string.
    """
    s = string
    # Strip meta <table> elements.
    s = strip_element(s, "table", "id=\"toc")             # Table of contents.
    s = strip_element(s, "table", "class=\"infobox")      # Infobox.
    s = strip_element(s, "table", "class=\"navbox")       # Navbox.
    s = strip_element(s, "table", "class=\"mbox")         # Message.
    s = strip_element(s, "table", "class=\"metadata")     # Metadata.
    s = strip_element(s, "table", "class=\".*?wikitable") # Table.
    s = strip_element(s, "table", "class=\"toc")          # Table (usually footer).
    # Strip meta <div> elements.
    s = strip_element(s, "div", "id=\"toc")               # Table of contents.
    s = strip_element(s, "div", "class=\"infobox")        # Infobox.
    s = strip_element(s, "div", "class=\"navbox")         # Navbox.
    s = strip_element(s, "div", "class=\"mbox")           # Message.
    s = strip_element(s, "div", "class=\"metadata")       # Metadata.
    s = strip_element(s, "div", "id=\"annotation")        # Annotations.
    s = strip_element(s, "div", "class=\"dablink")        # Disambiguation message.
    s = strip_element(s, "div", "class=\"magnify")        # Thumbnails.
    s = strip_element(s, "div", "class=\"thumb ")         # Thumbnail captions.
    s = strip_element(s, "div", "class=\"barbox")         # Bar charts.
    s = strip_element(s, "div", "class=\"mw-headline")    # Bar charts.
    s = strip_element(s, "div", "class=\"noprint")        # Hidden from print.
    s = strip_element(s, "sup", "class=\"noprint")
    # Strip absolute elements (don't know their position).
    s = strip_element(s, "div", "style=\"position:absolute")
    # Strip meta <span> elements.
    s = strip_element(s, "span", "class=\"error")
    # Strip math formulas, add [math] placeholder.
    s = re.sub(r"<img class=\"tex\".*?/>", "[math]", s)   # LaTex math images.
    s = plaintext(s)
    # Strip [edit] link (language dependent.)
    s = re.sub(r"\[edit\]\s*", "", s)
    s = re.sub(r"\[bewerken\]\s*", "", s)
    # Insert space before inline references.
    s = s.replace("[", " [").replace("  [", " [")
    # ignore lists?
    #s = " ".join([line.strip() for line in s.split('\n') if not line.strip().startswith('*')])
    return s

def neural_model(input_dim, hidden_dim, output_dim):
    # define standard two-layer model, with dropout+relu:
    model = Sequential()
    model.add(Dense(input_dim=input_dim, output_dim=hidden_dim, init="uniform"))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(input_dim=hidden_dim, output_dim=output_dim, init="uniform"))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Activation("softmax"))
    # compile model:
    model.compile(loss='categorical_crossentropy', optimizer='adadelta')
    return model

