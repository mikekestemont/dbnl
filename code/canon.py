from __future__ import print_function

from recsys.evaluation.ranking import SpearmanRho, KendallTau, MeanReciprocalRank

def get_canon(canon='kantl'):

    if canon not in ('kantl', 'mnl'):
        raise ValueError('Canon not available.')

    correct_authors = [] # not a set: preserve ranking!
    with open('../metadata_files/'+canon+'_canon.txt', 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                name, wiki_id = line.split('|')
                wiki_id = wiki_id.replace('https://nl.wikipedia.org/wiki/', '').strip()
                if wiki_id == "NA":
                    continue
                if wiki_id not in correct_authors:
                    correct_authors.append(wiki_id)

    return correct_authors

def overlap_score(correct_canon, predicted_canon):
    total = float(len(correct_canon))
    overlap = len([w for w in predicted_canon if w in correct_canon])
    return overlap/total

def kendall_tau(correct_canon, predicted_canon):
    """
    Can only consider the overlap between the two rankings...
    """
    correct_ranking = [(w, float(idx+1)) for idx, w in enumerate(correct_canon) if w in predicted_canon]
    predicted_ranking = [(w, float(idx+1)) for idx, w in enumerate(predicted_canon) if w in correct_canon]
    kendall = KendallTau()
    kendall.load(correct_ranking, predicted_ranking)
    return kendall.compute()

def mean_reciprocal_rank(correct_canon, predicted_canon):
    mrr = MeanReciprocalRank()
    for q in predicted_canon:
        mrr.load(correct_canon, q)
    return mrr.compute()



if __name__ == '__main__':
    kantl = get_canon('kantl')
    mnl = get_canon('mnl')
    print('kantl canon:', kantl)
    print('mnl canon:', mnl)
    print('overlap mnl > kantl:', overlap_score(correct_canon=mnl, predicted_canon=kantl))
    print('overlap kantl > mnl:', overlap_score(correct_canon=kantl, predicted_canon=mnl))
    print('tau kantl > mnl:', kendall_tau(correct_canon=kantl, predicted_canon=mnl))
    print('mrr kantl > mnl:', mean_reciprocal_rank(correct_canon=mnl, predicted_canon=kantl))
