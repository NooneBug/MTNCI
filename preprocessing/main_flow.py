# %%
from EntityNameRetriever import EntityNameRetriever
from graph import graph_from_edgelist, remove_void_types
import networkx as nx
from CorpusManager import CorpusManager
import pickle
from utils import save_data_with_pickle, load_data_with_pickle
import time
import random

# List of classes used to test the correctness of the workflow
# LIST_OF_CLASSES = ['City', 'Mosque', 'Animal']
# PATH in which utility files are stored
PICKLES_PATH = '../../source_files/pickles/'

# PATH that refers to the file which let the building of the Ontology Graph
PATH_TO_EDGELIST = PICKLES_PATH + 'dbpedia_edgelist_no_closure.tsv'
# PATH to the corpus from which information are extracted
CORPUS_PATH = '/datahdd/vmanuel/ELMo/Corpora/shuffled_text_with_words'
LOG = 'MEGAlog_3_3.txt'
avoid_multilabeling = True

# %%
if __name__ == "__main__":

    # %%

    # General design: if a resource is already built (in a previous iteration) load it, elsewhere build it and save it (for the future)

    # Create the ontology tree
    try:
        G = load_data_with_pickle(PICKLES_PATH + 'graph')
    except:
        G = graph_from_edgelist(PATH_TO_EDGELIST)
        save_data_with_pickle(PICKLES_PATH + 'graph', G)

    # Check if the built graph is a tree (it should be a tree because we need to use an Ontology Tree)
    print("the input graph is a tree: {}".format(nx.is_tree(G)))
    
    # %%
    
    # retrieve the ontology classes from the graph
    list_of_classes = [n for n in G.nodes()]

    # retrieve entities names
    e = EntityNameRetriever()
    
    FRAC = 0.02 # the time needed for the indexing part (logged) scale linearly with this variable, don't know why

    # random.seed(236451)
    try:
        1/0
        entity_dict = load_data_with_pickle(PICKLES_PATH + 'entity_dict')
    except:
        entity_dict = e.entities_from_types(random.sample(list_of_classes, int(FRAC*len(list_of_classes))))
        entity_dict = e.entity_name_preprocessing(entity_dict, max_words=20)
        save_data_with_pickle(PICKLES_PATH + 'entity_dict', entity_dict)

    # %%
    
    # entity_dict = {'Inventate': ['gatto', 'corona', 'virus']}
    # exclude from the tree the concepts which have no entities (only if the concept is a leaf or have all empty sons)
    void_types = [t for t, v in entity_dict.items() if v == []]

    pruned_G = remove_void_types(G, void_types)
    print("the pruned graph is a tree: {}".format(nx.is_tree(pruned_G)))

    # %%
    # Read a corpus, search the occurrences of the entity in the corpus, 
    # returns a dict in format {"entity_name": [list of tuples: (row, entity_name_occurrence_index)]}

    N = [2, 3, 4, 5, 6, 7, 10, 20]
    # N = [20]
    ENTITIES = [800, 1600, 3200, 6400, 12800, 25600]
    # LINES = [100000, 200000, 300000, 500000, 1000000, 5000000]
    LINES = [50000, 100000, 200000, 300000, 400000]

    with open(LOG, 'a') as out:
        out.write('\n\n')


    for l in LINES:
        c = CorpusManager()
        c.read_corpus(CORPUS_PATH, l)
        # c.joined_corpus = ['io sono un gatto', 
        #                     'sono in quarantena per il corona virus', 
        #                     'gatto gatto corona'
        #                     ] 
        c.create_all_entities(entity_dict, concepts=list_of_classes)    
        # c.create_all_entities(entity_dict, concepts=['Inventate'])    


        for e in ENTITIES:
            for n in N:
                word_indexes, n_proc, n_entities, leng, t = c.parallel_find(n_proc = n, n_entities= e)
                print('{:2} processes, {:5} entities, {:7} lines, {:8.2f} seconds'.format(n_proc, 
                                                                                n_entities, 
                                                                                leng,
                                                                                t
                                                                                ))
                
                with open(LOG, 'a') as out:
                    out.write('{:2} processes, {:5} entities, {:7} lines, {:8.2f} seconds\n'.format(n_proc, 
                                                                                n_entities, 
                                                                                leng,
                                                                                t
                                                                                ))
            with open(LOG, 'a') as out:
                out.write('\n')

        with open(LOG, 'a') as out:
            out.write('\n')

    save_data_with_pickle(PICKLES_PATH + 'word_occurrence_index', word_indexes)            

    print('{} entities found'.format(len(word_indexes.keys())))

    # # filter the entity dict maintaining only the entities found in the corpus
    # found_entities = set(word_indexes.keys())
    # found_entity_dict = {k: set(v).intersection(found_entities) for k,v in entity_dict.items() if set(v).intersection(found_entities)}
    
    # if avoid_multilabeling:
    #     found_entity_dict = c.avoid_multilabeling(found_entity_dict, G)

    # save_data_with_pickle(PICKLES_PATH + 'found_entity_dict', found_entity_dict)            

    


# %%
