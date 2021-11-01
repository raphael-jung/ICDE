import sqlite3
import networkx as nx
import pandas as pd
from itertools import islice
from itertools import chain
from gensim.models import KeyedVectors
from gensim import matutils
import numpy as np
import os

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

# Get neighbor informations to be processed
def record_fos_neighbors(years_to_record, number_of_pegonets_to_search):
    def get_graph(conn):
        c = conn.cursor()
        Gs = {}
        for year in years_to_record:
            # Get the graph for the given iteration.
            G = nx.Graph()
            # Get list of unique nodes which were used in the given year,
            c.execute(f'''
                SELECT DISTINCT PaperFieldsOfStudy.FieldOfStudyId, {year} - FirstYear
                FROM PaperFieldsOfStudy, FieldsOfStudy
                WHERE
                    PaperFieldsOfStudy.FieldOfStudyId = FieldsOfStudy.FieldOfStudyId AND
                    PaperFieldsOfStudy.Year = {year}
                ''')
            for n, age in c.fetchall():
                G.add_node(n, age=age)
            # Get list of fos combinations in the given year, with frequency as weight
            c.execute(f'''
                SELECT F1.FieldOfStudyId, F2.FieldOfStudyId, count(P.PaperId)
                FROM
                    (SELECT DISTINCT PaperId FROM PaperFieldsOfStudy WHERE Year = {year}) AS P,
                    PaperFieldsOfStudy AS F1,
                    PaperFieldsOfStudy AS F2
                WHERE
                    F1.PaperId = P.PaperId AND
                    F2.PaperId = P.PaperId AND
                    F1.FieldOfStudyId < F2.FieldOfStudyId
                GROUP BY F1.FieldOfStudyId, F2.FieldOfStudyId
                ''')
            for f, t, freq in c.fetchall():
                G.add_edge(f, t, weight=freq, distance=1.0/freq)
            Gs[year] = G
        return Gs

    def record_pegonet_properties(Gs, year, number_of_pegonets_to_search):
        def get_seeds(Gs, year, number_of_pegonets_to_search):

            # Divide new and old fos, at year+1
            seeds_new = sorted(((s, len(Gs[year+1][s].items())) for s in Gs[year+1].nodes() if Gs[year+1].nodes[s]['age'] == 0), key = lambda x: x[1], reverse=True)
            seeds_old = sorted(((s, len(Gs[year+1][s].items())) for s in Gs[year+1].nodes() if Gs[year+1].nodes[s]['age'] != 0), key = lambda x: x[1], reverse=True)

            sizes_new = [len(Gs[year+1][s].items()) for s in Gs[year+1].nodes() if Gs[year+1].nodes[s]['age'] == 0]
            sizes_old = [len(Gs[year+1][s].items()) for s in Gs[year+1].nodes() if Gs[year+1].nodes[s]['age'] != 0]

            # Filter the egonets to the desired number - there could be a large size discrepancies, so remove larger parts first before the actual filteration
            # If there are no egonets to get the size from, then use last year's max.
            newmax = max(sizes_new) if len(sizes_new) > 0 else max([len(Gs[year][s].items()) for s in Gs[year].nodes() if Gs[year].nodes[s]['age'] == 0])
            oldmax = max(sizes_old) if len(sizes_old) > 0 else max([len(Gs[year][s].items()) for s in Gs[year].nodes() if Gs[year].nodes[s]['age'] != 0])

            if newmax >= oldmax:
                new = islice(((s,c) for s,c in seeds_new if c <= oldmax), number_of_pegonets_to_search)
                old = islice(seeds_old, number_of_pegonets_to_search)
            else:
                new = islice(seeds_new, number_of_pegonets_to_search)
                old = islice(((s,c) for s,c in seeds_old if c <= newmax), number_of_pegonets_to_search)

            return chain(new, old)

        outcome = []
        seeds = get_seeds(Gs, year, number_of_pegonets_to_search)
        
        # Get Pegonets at year
        for seed, size_in_next_y in seeds:
            isnewfos = Gs[year+1].nodes[seed]['age'] == 0

            subgraph = nx.Graph(Gs[year].subgraph(Gs[year+1].neighbors(seed))) # the neighbor subgraph in y (the one which can actually be SEEN in the given year)
            subgraph_next_y = dict(Gs[year+1][seed].items()) # neighbor subgraph in y+1 (the one which actually happened with the topic) with weight and frequency to the seed node
            egonet = subgraph.nodes()

            # skip if the whole subgraph is just empty, or below the given size limit
            if len(subgraph.edges())==0 or len(subgraph.nodes())==0: continue

            outcome.append([year, seed, isnewfos, egonet, subgraph_next_y])
                
        df = pd.DataFrame(outcome, columns=['year','seed','isNewFos','neighbors','subgraph_next_year'])
        df.to_csv(fname_egonet_neighbors, index=False, mode='a', header=None)
        return None

    # Make connection to the dataset
    conn = sqlite3.connect(fname_fos)

    # Go over the given years to get the graph - 2000 ~ 2020 for now.
    Gs = get_graph(conn)

    # Go over all but the last year - the last year is not used because year+1 is used to identify pegonets!
    for year in years_to_record[:-1]:
        # Get egonets and store their properties
        record_pegonet_properties(Gs, year, number_of_pegonets_to_search)

    # Close the connection
    conn.close()

    return pd.read_csv(fname_egonet_neighbors)



# Location of the word 2 vector datas
w2v_list = [
    ('w2v_skip_hier', "D:\Dataset\medline-word-vector\word2vec_skip_hier.vec"),
    ('w2v_cbow_hier', "D:\Dataset\medline-word-vector\word2vec_cbow_hier.vec"),
    ('fst_skip_hier', "D:\Dataset\medline-word-vector\\fasttext_skip_hier.vec")]

# Location of fos file and the target domain
fname_fosname = "D:\Dataset\MAG 2020-02-14 from Azure\FieldsOfStudy\\0.db"
fname_simvec = f"./seed_neighbor_sim.csv"
fname_skippedfos = f"./skipped-seed.csv"
df_simvec = pd.read_csv(fname_simvec)

yrange = list(range(2001,2017))
egosize= 500

domain_list = [f.split('__')[0]+'__'+f.split('.')[0].split('__')[1].split('_')[0] for f in os.listdir("E:\Dataset\MAG_medicine\egonets")]
# domain_list = ['optometry__1']

# Read word2vec
for mname, fname in w2v_list:
    word_vecs = KeyedVectors.load_word2vec_format(fname, binary=False)

    for domain in domain_list:
        fname_fos     = 'E:\Dataset\MAG_toplevel\\'+[i for i in os.listdir('E:\Dataset\MAG_toplevel') if os.path.isfile(os.path.join('E:\Dataset\MAG_toplevel',i)) and f"mag-{domain}" in i][0]
        fname_egonet_neighbors = f"E:\Dataset\MAG_medicine\\neighbors\\{domain}.csv"
        
        # Then check if the result is already made into the output file - skip those that are already done.
        if len(df_simvec.loc[(df_simvec['domain']==domain.split('__')[0]) & (df_simvec['trained_model']==mname)]) > 0:
            print(domain, mname, "combination already done, skipping")
            continue
        else:
            print(domain, mname, "combination undergoing")

        # First, get the neighbor informations to a file
        # Either populate then get df, or just existing df
        if os.path.isfile(fname_egonet_neighbors):
            df_neighbors = pd.read_csv(fname_egonet_neighbors)
        else:
            # create file with headers
            with open(fname_egonet_neighbors, 'w') as f:
                f.write('year,seed,isNewFos,neighbors,subgraph_next_year\n')
            df_neighbors = record_fos_neighbors(yrange, egosize)
        
        # then, get the fos names that'll be used for the vector calculation
        dict_names = get_fos_names()

        # from: https://groups.google.com/g/gensim/c/OfBjD0GU2xA?pli=1
        # loop through each seed
        outcome = []
        skipped_output = []
        for _, row in df_neighbors.iterrows():
            try:
                target_sent   = [w for w in dict_names[row['seed']].lower().split() if (w in word_vecs) & (w != None) ]
                neighbor_sent = []
                for i in eval(row['neighbors']):
                    if (i in dict_names)  & (dict_names[i] != None):
                        for j in dict_names[i].lower().split():
                            if j in word_vecs:
                                neighbor_sent.append(j)
            except Exception as e:
                print("something went wrong getting the sentences", row['seed'], row['year'], dict_names[row['seed']], [dict_names[i] for i in eval(row['neighbors'])])
                continue

            # See if the seed is found.
            if len(target_sent) == 0:
                skipped_output.append([domain.split('__')[0], mname, row['year'], dict_names[row['seed']] ,row['isNewFos']])
                continue

            # calculate sim value
            sim = word_vecs.n_similarity(target_sent , neighbor_sent)

            # get most similar words (top 3, with numbers)
            similar_words = word_vecs.most_similar(positive=neighbor_sent, topn=5)

            outcome.append([domain.split('__')[0], mname, row['year'],dict_names[row['seed']],row['isNewFos'], sim, similar_words])
            
        pd.DataFrame(skipped_output, columns=['domain', 'trained_model', 'year', 'seed', 'isNewFos']).to_csv(fname_skippedfos, index=False, mode='a', header=None)
        pd.DataFrame(outcome       , columns=['domain', 'trained_model', 'year', 'seed', 'isNewFos', 'sim', 'most_simialr']).to_csv(fname_simvec, index=False, mode='a', header=None)