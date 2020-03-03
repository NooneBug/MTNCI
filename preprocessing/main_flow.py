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
LIST_OF_CLASSES = ['City', 'Mosque', 'Animal']
# PATH in which utility files are stored
PICKLES_PATH = '../../source_files/pickles/'

# PATH that refers to the file which let the building of the Ontology Graph
PATH_TO_EDGELIST = PICKLES_PATH + 'dbpedia_edgelist_no_closure.tsv'
# PATH to the corpus from which information are extracted
CORPUS_PATH = '/datahdd/vmanuel/ELMo/Corpora/shuffled_text_with_words'


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
    
    FRAC = 0.25 # the time needed for the indexing part (logged) scale linearly with this variable, don't know why

    try:
        1/0
        entity_dict = load_data_with_pickle(PICKLES_PATH + 'entity_dict')
    except:
        entity_dict = e.entities_from_types(random.sample(list_of_classes, int(FRAC*len(list_of_classes))))
        entity_dict = e.entity_name_preprocessing(entity_dict, max_words=20)
        save_data_with_pickle(PICKLES_PATH + 'entity_dict', entity_dict)

    # %%
    # exclude from the tree the concepts which have no entities (only if the concept is a leaf or have all empty sons)
    void_types = [t for t, v in entity_dict.items() if v == []]

    pruned_G = remove_void_types(G, void_types)
    print("the pruned graph is a tree: {}".format(nx.is_tree(pruned_G)))

    # %%
    # Read a corpus, search the occurrences of the entity in the corpus, 
    # returns a dict in format {"entity_name": [list of tuples: (row, entity_name_occurrence_index)]}

    # try:
    #     # 1/0
    #     print('load corpus...')
    #     t = time.time()
    #     c.corpus = load_data_with_pickle(PICKLES_PATH + 'corpus')
    #     print('corpus loaded in {} seconds'.format(round(time.time() - t, 2)))
    #     # print('load vocab...')
    #     # c.vocab = load_data_with_pickle(PICKLES_PATH + 'vocab')
    #     # print('load entities...')
    #     # c.concept_entities = load_data_with_pickle(PICKLES_PATH + 'concept_entities')
    # except:
    #     # c.check_words(entity_dict)
    #     save_data_with_pickle(PICKLES_PATH + 'corpus', c.corpus)
    #     # save_data_with_pickle(PICKLES_PATH + 'vocab', c.vocab)
    #     # save_data_with_pickle(PICKLES_PATH + 'concept_entities', c.concept_entities)

    # %%
    N = [10, 15, 20, 25, 30]
    ENTITIES = [200]
    # LINES = [100000, 200000, 300000, 500000, 1000000, 5000000]
    LINES = [1000]
    with open('log.txt', 'a') as out:
            out.write('\n\nentities with only two words\n')
    for l in LINES:
        c = CorpusManager()
        c.read_corpus(CORPUS_PATH, l)
        c.create_all_entities(entity_dict, concepts=list_of_classes)    
        for e in ENTITIES:
            for n in N:
                f, n_proc, n_entities, leng, t = c.parallel_find(n_proc = n, n_entities= e)
                t2 = time.time()
                print('{:2} processes, {:5} entities, {} lines, {:8.2f} seconds'.format(n_proc, 
                                                                                n_entities, 
                                                                                leng,
                                                                                t2 - t
                                                                                ))
                
                with open('log.txt', 'a') as out:
                    out.write('{:2} processes, {:5} entities, {} lines, {:8.2f} seconds\n'.format(n_proc, 
                                                                                n_entities, 
                                                                                leng,
                                                                                t2 - t
                                                                                ))
        
        with open('log.txt', 'a') as out:
            out.write('\n')
    save_data_with_pickle(PICKLES_PATH + 'word_occurrence_index', f)            
          
    # t = time.time()
    # print('{} entities found'.format(len(f)))
    


# %%
