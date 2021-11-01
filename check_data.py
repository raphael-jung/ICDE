import sqlite3
import networkx as nx
import pandas as pd
from itertools import islice
from itertools import chain
from gensim.models import KeyedVectors
from gensim import matutils
import numpy as np
import os

# Location of the word 2 vector datas
w2v_list = [
    ('w2v_skip_hier', "D:\Dataset\medline-word-vector\word2vec_skip_hier.vec"),
    ('w2v_cbow_hier', "D:\Dataset\medline-word-vector\word2vec_cbow_hier.vec"),
    ('fst_skip_hier', "D:\Dataset\medline-word-vector\\fasttext_skip_hier.vec")]

# Location of fos file and the target domain
fname_fosname = "D:\Dataset\MAG 2020-02-14 from Azure\FieldsOfStudy\\0.db"
fname_simvec = f"./seed_neighbor_sim_meansubtracted.csv"
fname_skippedfos = f"./skipped-seed_meansubtracted.csv"
df_simvec = pd.read_csv(fname_simvec)

yrange = (2011,2017)
egosize= 500

domain_list = [f.split('__')[0]+'__'+f.split('.')[0].split('__')[1].split('_')[0] for f in os.listdir("E:\Dataset\MAG_medicine\egonets")]
# domain_list = ['optometry__1']



def get_fos_names():
    # connect to the domain dataet
    conn = sqlite3.connect(fname_fos)
    c = conn.cursor()

    # Connect fos db
    c.execute(f"ATTACH '{fname_fosname}' AS dbn")

    # get the fos/names
    r = c.execute("""
        SELECT FosId, FosName
        FROM FieldsOfStudy INNER JOIN dbn.FosName
        ON FieldsOfStudy.FieldOfStudyId = dbn.FosName.FosId
        """).fetchall()
    dic = {i[0]:i[1] for i in r}
    # close the connection
    conn.close()

    return dic

# Read word2vec
for mname, fname in w2v_list:
    word_vecs = KeyedVectors.load_word2vec_format(fname, binary=False)

    for domain in domain_list:
        fname_fos = 'E:\Dataset\MAG_toplevel\\'+[i for i in os.listdir('E:\Dataset\MAG_toplevel') if os.path.isfile(os.path.join('E:\Dataset\MAG_toplevel',i)) and f"mag-{domain}" in i][0]
        fname_egonet_neighbors = f"E:\Dataset\MAG_medicine\\neighbors\\{domain}.csv"

        # Get neighbors
        df_neighbors = pd.read_csv(fname_egonet_neighbors)
        df_neighbors = df_neighbors.loc[(df_neighbors['year'] >= yrange[0]) & (df_neighbors['year'] < yrange[1])]
        

        # then, get the fos names that'll be used for the vector calculation
        dict_names = get_fos_names()

        # get all the words that are used AND are in the word vectors
        words_freq = [w for v in dict_names.values() if v != None for w in v.split(' ') if w in word_vecs ]
        words = set(words_freq)


        # get mean of whole vectors - in unitvec form too
        mean_vec = np.mean(word_vecs[words], axis=0)
        mean_unitvec = matutils.unitvec(mean_vec)
        mean_vec_freq = np.mean(word_vecs[words_freq], axis=0)
        mean_unitvec_freq = matutils.unitvec(mean_vec_freq)

        outcome = []
        skipped_output = []
        for _, row in df_neighbors.iterrows():
            target_sent = []
            try:
                neighbor_sent = []
                for i in eval(row['neighbors']):
                    if (i in dict_names)  & (dict_names[i] != None):
                        for j in dict_names[i].lower().split():
                            if j in word_vecs:
                                neighbor_sent.append(j)
                target_sent   = [w for w in dict_names[row['seed']].lower().split() if (w in word_vecs) & (w != None) ]
            except Exception as e:
                # print("something went wrong getting the sentences", domain.split('__')[0], mname, row['year'], dict_names[row['seed']])
                skipped_output.append([domain.split('__')[0], mname, row['year'], dict_names[row['seed']] ,row['isNewFos']])
                continue

            # See if the seed is found.
            if len(target_sent) < len(dict_names[row['seed']].lower().split()) :
                # print("target topic not found in word vector", domain.split('__')[0], mname, row['year'], dict_names[row['seed']])
                skipped_output.append([domain.split('__')[0], mname, row['year'], dict_names[row['seed']] ,row['isNewFos']])
                continue

            # Calculate sim value when mean is not subtracted
            sim = word_vecs.n_similarity(target_sent , neighbor_sent)

            # calculate sim value - when mean is subtracted before applying unitvec
            v1 = np.mean(word_vecs[target_sent], axis=0)
            v2 = np.mean(word_vecs[neighbor_sent], axis=0)
            sim2 = np.dot(
                matutils.unitvec(v1 - mean_vec),
                matutils.unitvec(v2 - mean_vec))

            # calculate sim value - when unitvec mean is subtracted from unitvecs (and then again unitveced for dot product)
            sim3 = np.dot(
                matutils.unitvec(matutils.unitvec(v1) - mean_unitvec),
                matutils.unitvec(matutils.unitvec(v2) - mean_unitvec))

            # FREQ calculate sim value - when mean is subtracted before applying unitvec
            sim4 = np.dot(
                matutils.unitvec(v1 - mean_vec_freq),
                matutils.unitvec(v2 - mean_vec_freq))

            # FREQ calculate sim value - when unitvec mean is subtracted from unitvecs
            sim5 = np.dot(
                matutils.unitvec(matutils.unitvec(v1) - mean_unitvec_freq),
                matutils.unitvec(matutils.unitvec(v2) - mean_unitvec_freq))

            # Get dot product variants instead
            dot = np.dot(v1, v2)
            dot2 = np.dot(v1 - mean_vec, v2 - mean_vec)
            dot3 = np.dot(v1 - mean_vec_freq, v2 - mean_vec_freq)


            outcome.append([domain.split('__')[0], mname, row['year'], dict_names[row['seed']], row['isNewFos'], sim, sim2, sim3, sim4, sim5, dot, dot2, dot3])
                        
        pd.DataFrame(skipped_output, columns=['domain', 'trained_model', 'year', 'seed', 'isNewFos']).to_csv(fname_skippedfos, index=False, mode='a', header=None)
        pd.DataFrame(outcome       , columns=['domain', 'trained_model', 'year', 'seed', 'isNewFos', 'sim_default', 'sim_sub', 'sim_unitsub', 'sim_sub_freq', 'sim_unitsub_freq', 'dot_default', 'dot_sub', 'dot_sub_freq']).to_csv(fname_simvec, index=False, mode='a', header=None)
