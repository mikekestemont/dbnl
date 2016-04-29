import pickle
import unicodedata
import os
import codecs
import csv
import glob
import re

from operator import itemgetter
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sb

import pandas as pd
import numpy as np
from bs4 import BeautifulSoup

from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering

from bokeh.models import HoverTool, ColumnDataSource
from bokeh.plotting import figure, show, output_file, save
from bokeh.charts import Bar
from bokeh.io import output_file, show, vplot, save
from bokeh.plotting import figure
from bokeh.models import Axis
from bokeh.models.ranges import FactorRange
from bokeh.charts.attributes import ColorAttr, CatAttr

from gensim.models import Word2Vec

from feature_extraction import DirectoryIterator
import canon

def author_frequency_barplot(nb_top_ners=50, tf=True):
    if tf:
        output_file('../figures/tf_authors.html')
        ner_freqs = pickle.load(open('../workspace/tf.m', "rb"))
    else:
        output_file('../figures/df_authors.html')
        ner_freqs = pickle.load(open('../workspace/df.m', "rb"))

    top_ners = [w for w,_ in ner_freqs.most_common(nb_top_ners)]
    top_freqs = [c for _,c in ner_freqs.most_common(nb_top_ners)]

    names = []
    for name in top_ners:
        name = name.replace('*', '')
        if '(' in name:
            name = name.split('(')[0].strip()
        name = ' '.join([n.lower().capitalize() for n in name.split('_')])
        name = ''.join([c for c in unicodedata.normalize('NFKD', name)
                if not unicodedata.combining(c)])
        names.append(name)

    data = pd.DataFrame({'values': top_freqs[:25], 'labels': names[:25]})
    bar1 = Bar(data, label=CatAttr(columns=['labels'], sort=False), values='values', title='Author Frequency (1-25)', width=800, height=400)
    xaxis = bar1.select(dict(type=Axis))[1]
    xaxis.major_label_standoff = 0
    xaxis.major_label_orientation = np.pi/2
    xaxis.major_label_standoff = 6
    xaxis.major_tick_out = 0

    data = pd.DataFrame({'values': top_freqs[25:50], 'labels': names[25:50]})
    bar2 = Bar(data, label=CatAttr(columns=['labels'], sort=False), values='values', title='Author Frequency (25-50)', width=800, height=400)
    xaxis = bar2.select(dict(type=Axis))[1]
    xaxis.major_label_standoff = 0
    xaxis.major_label_orientation = np.pi/2
    xaxis.major_label_standoff = 6
    xaxis.major_tick_out = 0
    p = vplot(bar1, bar2)
    save(p)

def w2v_author_similarities():
    w2v_model = Word2Vec.load(os.path.abspath('../workspace/w2v_model.m'))
    
    print("Claus:", w2v_model.most_similar('*Hugo_Claus', topn=25))
    print("Maerlant:", w2v_model.most_similar('*Jacob_van_Maerlant', topn=25))
    #print(w2v_model.most_similar('*JACOB_VAN_MAERLANT', topn=25))
    
    print("Roman:", [w for w,v in w2v_model.most_similar('roman', topn=1000) if w.startswith('*')])
    print("Gedicht:", [w for w,v in w2v_model.most_similar('gedicht', topn=1000) if w.startswith('*')])
    print("vijftigers:", [w for w,v in w2v_model.most_similar('vijftigers', topn=1000) if w.startswith('*')])
    print("tachtigers:", [w for w,v in w2v_model.most_similar('tachtigers', topn=1000) if w.startswith('*')])
    
    # plus: 
    print([w for w,v in w2v_model.most_similar(
        positive=['*Harry_Mulisch', '*Willem_Frederik_Hermans', '*Gerard_Reve'],
        topn=1000) if w.startswith('*')][:10])
    print([w for w,v in w2v_model.most_similar(
        positive=['*Hugo_Claus', '*Louis_Paul_Boon'],
        topn=1000) if w.startswith('*')][:10])
    print([w for w,v in w2v_model.most_similar(
        positive=['*Harry_Mulisch', '*Willem_Frederik_Hermans', '*Gerard_Reve'],
        negative=['*Hugo_Claus', '*Louis_Paul_Boon'],
        topn=1000) if w.startswith('*')][:10])
    print([w for w,v in w2v_model.most_similar(
        negative=['*Harry_Mulisch', '*Willem_Frederik_Hermans', '*Gerard_Reve'],
        positive=['*Hugo_Claus', '*Louis_Paul_Boon'],
        topn=1000) if w.startswith('*')][:10])
    
    print([w for w,v in w2v_model.most_similar(
            negative=['mannen', 'man', 'jongen', 'jongens'],
            positive=['vrouw', 'vrouwen', 'meisje', 'meisjes'], topn=25)])
    print([w for w,v in w2v_model.most_similar(
            positive=['mannen', 'man', 'jongen', 'jongens'],
            negative=['vrouw', 'vrouwen', 'meisje', 'meisjes'], topn=25)])
    print([w for w,v in w2v_model.most_similar(
            negative=['dichter', 'schrijver'],
            positive=['dichteres', 'schrijfster'], topn=25)])
    print([w for w,v in w2v_model.most_similar(
            positive=['dichter', 'schrijver'],
            negative=['dichteres', 'schrijfster'], topn=25)])
    print([w for w,v in w2v_model.most_similar(
            positive=['*Herman_Gorter', 'vijftigers'],
            negative=['tachtigers'], topn=25)])
    print([w for w,v in w2v_model.most_similar(
            positive=['*Harry_Mulisch', 'gedicht'],
            negative=['roman'], topn=25)])

def author_tsne_plot():
    canonized = set()
    for name in canon.get_canon('kantl'):
        name = name.replace('*', '')
        if '(' in name:
            name = name.split('(')[0].strip()
        name = " ".join([n.lower().capitalize() for n in name.split('_')])
        canonized.add(name)
    print(canonized)
    
    ner_freqs = pickle.load(open('../workspace/tf.m', "rb"))
    top_ners = ['*'+t for t,q in ner_freqs.most_common(500)]
    print(top_ners)
    w2v_model = Word2Vec.load(os.path.abspath('../workspace/w2v_model.m'))
    full_matrix = np.asarray([w2v_model[w] for w in top_ners if w in w2v_model], dtype='float64')
    print(full_matrix.shape)
    tsne = TSNE(n_components=2,
                random_state=1987,
                verbose=1,
                n_iter=2500,
                perplexity=4.0,
                early_exaggeration=4.0,
                learning_rate=1000)
    tsne_projection = tsne.fit_transform(full_matrix)
    
    names = []
    for name in top_ners:
        if name in w2v_model:
            name = name.replace('*', '')
            if '(' in name:
                name = name.split('(')[0].strip()
            name = " ".join([n.lower().capitalize() for n in name.split('_')])
            names.append(name)
    
    # clustering on top (for reading-aid coloring):
    clusters = AgglomerativeClustering(n_clusters=8).fit_predict(tsne_projection)
    
    # get color palette:
    colors = sb.color_palette('husl', n_colors=8)
    colors = [tuple([c * 256 for c in color]) for color in colors]
    colors = ['#%02x%02x%02x' % colors[i] for i in clusters]
    
    TOOLS="pan,wheel_zoom,reset,hover,box_select,save"
    source = ColumnDataSource(data=dict(x=tsne_projection[:,0], y=tsne_projection[:,1], name=names))
    output_file('../figures/bokeh_embeddings.html')
    p = figure(title='Author embeddings space',
               tools=TOOLS,
               plot_width=1000,
               title_text_font="Arial", 
               plot_height=800,
               outline_line_color="white")
    p.circle(x=tsne_projection[:,0],
             y=tsne_projection[:,1],
             source=source,
             size=8,
             color=colors,
             fill_alpha=0.9,
             line_color=None)
    
    for name, x, y in zip(names, tsne_projection[:,0], tsne_projection[:,1]):
        if name in canonized:
            p.text(x, y, text=[name], text_align="center", text_font_size="10pt")
    
    hover = p.select(dict(type=HoverTool))
    hover.tooltips = [("name", "@name")]
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.major_label_text_font_size = '0pt'
    
    # Turn off tick marks
    p.axis.major_tick_line_color = None
    p.axis[0].ticker.num_minor_ticks = 0
    p.axis[1].ticker.num_minor_ticks = 0
    save(p)

def temporal_lda_plot(max_documents=50000000,
                      max_words_per_doc=50000000,
                      n_topics=100):
    # load the model:
    lda_model = pickle.load(open('../workspace/lda_model.m', 'rb'))
    # get top words per topic for title above plots:
    top_words = []
    for topic in lda_model.show_topics(num_topics=100, num_words=15, formatted=False):
        top_words.append(" ".join([w for _, w in topic]))
    
    # extract years from in the file ids in the original file:
    text_ids = list(DirectoryIterator(path_pattern='../workspace/wikified_periodicals/*.wikified',
                                     max_documents=max_documents,
                                     max_words_per_doc=max_words_per_doc,
                                     get='filename'))
    years = [id_.split('-')[-1].replace('.wikified', '') for id_ in text_ids]
    year_scores = {y:list() for y in years}
    
    # extract topic scores for each document:
    for line, year in zip(codecs.open('../workspace/mallet_output/doctopics.txt', 'r', 'utf8'), years):
        line = line.strip()
        if not line:
            continue
        comps = line.strip().split()
        topic_scores = [float(t) for t in comps[2:]]
        year_scores[year].append(topic_scores)

    for year in year_scores:
        m = np.asarray(year_scores[year], dtype='float32')
        year_scores[year] = np.mean(m, axis=0)

    year_scores = [(int(year),matrix) for year, matrix in year_scores.items() if int(year) <= 2010]
    year_scores = sorted(year_scores, key=itemgetter(0))

    output_file("../figures/topics.html")
    plots = []

    year_labels = [year for year, _ in year_scores if int(year) <= 2010]

    for topic_idx in range(n_topics):
        scores = [m[topic_idx] for y, m in year_scores]

        p = figure(title=top_words[topic_idx], plot_width=1200, plot_height=400, title_text_font_size='12pt')
        p.line(year_labels, scores, line_width=2)
        plots.append(p)

    p = vplot(*plots)
    save(p)


def plot_nb_texts(metadata_dict={}, text_names=None):
    print('orig nb items:', len(metadata_dict))

    if text_names:
        # we only do this for the provided files
        metadata_dict = {k:v for k,v in metadata_dict.items() if k in text_names}
    
    print('actual nb items:', len(metadata_dict))
    
    # plot main genres:
    genre_cnt = Counter((metadata_dict[text]['genre'] for text in metadata_dict if 'genre' in metadata_dict[text]))
    genre_items = sorted(genre_cnt.items(), key=itemgetter(1), reverse=True)
    genres, genre_cnts = zip(*genre_items)

    y_pos = np.arange(len(genres))
    plt.barh(y_pos, genre_cnts, align="center")
    plt.yticks(y_pos, genres)
    plt.title('# teksten per hoofdcategorie')
    plt.tight_layout()
    plt.savefig('../figures/genre_nb.pdf')
    plt.clf()

    # plot subgenres:
    subgenre_cnt = Counter((metadata_dict[text]['subgenre'] for text in metadata_dict if 'subgenre' in metadata_dict[text]))
    subgenre_items = sorted(subgenre_cnt.items(), key=itemgetter(1), reverse=True)[:15] # only plot the 10 most common ones
    subgenres, subgenre_cnts = zip(*subgenre_items)
    subgenres = list([BeautifulSoup(s, 'lxml').text for s in subgenres])

    y_pos = np.arange(len(subgenres))
    plt.barh(y_pos, subgenre_cnts, align='center')
    plt.yticks(y_pos, subgenres)
    plt.title('# teksten per subcategorie')
    plt.tight_layout()
    plt.savefig('../figures/subgenre_nb.pdf')
    plt.clf()

def plot_size_texts(metadata):
    genres = ("proza", "poëzie", "drama", "sec - letterkunde", "sec - taalkunde", 
          "sec - taal/letterkunde", "jeugdliteratuur", "_",  "non-fictie")
    genre_sizes = {k:0 for k in genres}

    subgenre_sizes = {}
    with open("../metadata_files/subgenre.txt", 'r') as csvfile:
        for row in csv.reader(csvfile, delimiter=';', quotechar='"'):
            try:
                _, subgenre_id = row
                subgenre_sizes[subgenre_id] = 0
            except UnicodeDecodeError:
                pass
    
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
    y_pos = np.arange(len(genres))
    plt.barh(y_pos, genre_cnts, align="center")
    plt.yticks(y_pos, genres)
    plt.title('Datagrootte per hoofdcategorie')
    plt.tight_layout()
    plt.savefig('../figures/genre_size.pdf')
    plt.clf()

    subgenre_sizes = {k:(v/float(1000000000)) for k,v in subgenre_sizes.items() if v} # convert to Megabytes
    subgenre_items = sorted(subgenre_sizes.items(), key=itemgetter(1), reverse=True)[:15] # only plot the 10 most common ones
    subgenres, subgenre_cnts = zip(*subgenre_items)
    subgenres = list([BeautifulSoup(s, 'lxml').text for s in subgenres])

    y_pos = np.arange(len(subgenres))
    plt.barh(y_pos, subgenre_cnts, align='center')
    plt.yticks(y_pos, subgenres)
    plt.title('Datagrootte per subcategorie')
    plt.tight_layout()
    plt.savefig('../figures/subgenre_size.pdf')
    plt.clf()
    

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

def plot_periodicals(metadata):
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
        title = BeautifulSoup(title, 'lxml').text
        print(title)
        if len(title) > 30:
            title = "".join([w[0] for w in title.split() if w[0].isupper()])
            print(title)
        clean_names.append(title)
    
    y_pos = np.arange(len(clean_names))
    plt.barh(y_pos, journal_cnts, align='center')
    plt.yticks(y_pos, clean_names)
    plt.title('Cumulatief # tijdschriftnummers')
    plt.tight_layout()
    plt.savefig('../figures/journal_issues.pdf')
    plt.clf()

def plot_frogged_data(metadata):
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
        if iter_cnt % 500 == 0:
            print(iter_cnt, 'documents parsed')
        #if iter_cnt >= 1000:
        #    break

    print('total nb of words in corpus', sum(words_per_year.values()))
    print('total nb of articles in corpus', sum(texts_per_year.values()))

    word_counts = {k:v for k,v in words_per_year.items() if v} # convert to megabytes
    word_items = sorted(word_counts.items(), key=itemgetter(0), reverse=False)
    years, word_cnts = zip(*word_items)

    y_pos = np.arange(len(years))
    plt.barh(y_pos, word_cnts, align='center')
    plt.yticks(y_pos, years)
    plt.title('# woorden per jaar (1945-2010)')
    plt.tight_layout()
    plt.savefig('../figures/words_per_year.pdf')
    plt.clf()

    text_counts = {k:v for k,v in texts_per_year.items() if v} # convert to megabytes
    text_items = sorted(text_counts.items(), key=itemgetter(0), reverse=False)
    years, text_cnts = zip(*text_items)

    y_pos = np.arange(len(years))
    plt.barh(y_pos, years, align='center')
    plt.yticks(y_pos, text_cnts)
    plt.title("# 'artikels' per jaar (1945-2010)")
    plt.tight_layout()
    plt.savefig('../figures/articles_per_year.pdf')
    plt.clf()

