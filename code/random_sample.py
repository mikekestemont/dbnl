# Select random sample (1 article per year: 1946-2010) for manual annotation
# Needed to evaluate the wikifier on.

import os
import glob
import shutil
import random
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sb

random.seed('456965')

year_dict = {}
file_stats = {}
cnt = 0
for filename in sorted(glob.glob('../workspace/wikified_periodicals/*.wikified')):
    cnt += 1
    if cnt % 1000 == 0:
        print(cnt)

    nb_words, nb_nes = 0.0, 0.0
    for line in open(filename, 'r'):
        try:
            comps = [c for c in line.strip().split('\t') if c]
            idx, token, lemma, pos, conf, ne, wiki = comps
            nb_words += 1
            if ne.startswith('B-PER') and token[0].isupper() and len(token) > 3 and not token.endswith('.'):
                nb_nes += 1
        except ValueError:
            continue
    file_stats[filename] = (nb_words, nb_nes)
    year = os.path.basename(filename).replace('.wikified', '').split('-')[-1]
    try:
        year_dict[year].add(filename)
    except KeyError:
        year_dict[year] = set()
        year_dict[year].add(filename)

nb_words = np.array([file_stats[k][0] for k in file_stats])
prop_nes = np.array([file_stats[k][1]/(file_stats[k][0]) for k in file_stats if (file_stats[k][0] and file_stats[k][1])])
sb.distplot(nb_words)
sb.plt.savefig('word_distr.pdf')
sb.plt.clf()
sb.distplot(prop_nes)
sb.plt.savefig('nes_distr.pdf')

print(np.mean(nb_words))

long_enough = set()
for k in file_stats:
    if file_stats[k][0] > np.mean(nb_words) and file_stats[k][1] and file_stats[k][0] and (file_stats[k][1]/file_stats[k][0] > 0.03):
        long_enough.add(k)
print(list(long_enough)[:10])

print(len(long_enough))

random_selection = set()
for year in sorted(year_dict):
    try:
        options = tuple(year_dict[year])
        options = [opt for opt in options if opt in long_enough]
        winner = random.choice(options)
        random_selection.add(winner)
        print(winner)
    except:
        continue

selection_path = '../workspace/random_sample/'
if os.path.isdir(selection_path):
    shutil.rmtree(selection_path)
os.mkdir(selection_path)

for o in random_selection:
    b = os.path.basename(o)
    shutil.copy(o, os.sep.join((selection_path, b)))
