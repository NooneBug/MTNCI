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
LOG = 'MEGAlog_4_3.txt'

WORD_OCCURRENCE_INDEX_PATH = PICKLES_PATH + 'word_occurrence_index_32150_100000static'
LENGTH = 100000

avoid_multilabeling = True
experimental = False

def experimental_run(list_of_classes):
    # N = [1]
    ENTITIES = [0.25, 0.5, 0.75, 1]
    # LINES = [5000]
    LINES = [500000, 1000000, 5000000]

    with open(LOG, 'a') as out:
        out.write('\n\n')

    for l in LINES:
        N = [i for i in range(15, 36)]
        c = CorpusManager()
        c.read_corpus(CORPUS_PATH, l)
        # c.joined_corpus = ['io sono un gatto', 
        #                     'sono in quarantena per il corona virus', 
        #                     'gatto gatto corona'
        #                     ] 
        c.create_all_entities(entity_dict, concepts=list_of_classes)    
        # c.create_all_entities(entity_dict, concepts=['Inventate'])    


        for e in ENTITIES:
            prev_t = 100000000
            i = 0
            while i < len(N):
                n = N[i]
                word_indexes, n_proc, n_entities, leng, t = c.parallel_find(n_proc = n, n_entities= e)
                print('{:2} processes, {:^5}/{:^7} entity-tokens, {:7} lines, {:8.2f} seconds'.format(n_proc, 
                                                                                n_entities,
                                                                                len(c.entities_token_in_corpus),
                                                                                leng,
                                                                                t
                                                                                ))
                
                with open(LOG, 'a') as out:
                    out.write('{:2} processes, {:^5}/{:^7} entity-tokens, {:7} lines, {:8.2f} seconds\n'.format(n_proc, 
                                                                                n_entities, 
                                                                                len(c.entities_token_in_corpus),
                                                                                leng,
                                                                                t
                                                                                ))
                
                if prev_t == 100000000:
                    prev_t = t
                    i += 1
                else:
                    if t > prev_t:
                        print('break')
                        with open(LOG, 'a') as out:
                            out.write('break\n')
                        break
                    else:
                        prev_t = t

                        # print('n = {}, delete {}'.format(n, n - 1))
                        # print('N before: {}'.format(N)) 
                        del N[N.index(n) - 1] 
                        # print('N after: {}'.format(N))

            print('{} entities found'.format(len(word_indexes.keys())))
            save_data_with_pickle(PICKLES_PATH + 'word_occurrence_index_{}_{}'.format(n_entities, leng), word_indexes)            
            with open(LOG, 'a') as out:
                out.write('{} entity-tokens found\n'.format(len(word_indexes.keys())))

        with open(LOG, 'a') as out:
            out.write('\n')

    occurrences_of_entities = c.check_composite_words(word_indexes = word_indexes, entity_dict = entity_dict)
    save_data_with_pickle(PICKLES_PATH + 'occurrences_of_entities_all', occurrences_of_entities)

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
    
    FRAC = 1 
    # random.seed(236451)
    try:
        entity_dict = load_data_with_pickle(PICKLES_PATH + 'entity_dict_9_3')
    except:
        entity_dict = e.entities_from_types(random.sample(list_of_classes, int(FRAC*len(list_of_classes))))
        entity_dict = e.entity_name_preprocessing(entity_dict, max_words=20)
        save_data_with_pickle(PICKLES_PATH + 'entity_dict_9_3', entity_dict)

    # %%
    
    # exclude from the tree the concepts which have no entities (only if the concept is a leaf or have all empty sons)
    void_types = [t for t, v in entity_dict.items() if v == []]

    pruned_G = remove_void_types(G, void_types)
    print("the pruned graph is a tree: {}".format(nx.is_tree(pruned_G)))

    # %%
    # Read a corpus, search the occurrences of the entity in the corpus, 
    # returns a dict in format {"entity_name": [list of tuples: (row, entity_name_occurrence_index)]}

    if experimental:
        experimental_run(list_of_classes = list_of_classes)
        
    else:
        c = CorpusManager()
        c.read_corpus(CORPUS_PATH, LENGTH)
        try:
            1/0
            word_indexes = load_data_with_pickle(WORD_OCCURRENCE_INDEX_PATH)
            occurrences_of_entities = c.check_composite_words(word_indexes = word_indexes, entity_dict = entity_dict)

            save_data_with_pickle(PICKLES_PATH + 'occurrences_of_entities', occurrences_of_entities)
        except:
            print('load occurrence of entities')
            occurrences_of_entities = load_data_with_pickle(PICKLES_PATH + 'occurrences_of_entities_9_3')
    
    # # filter the entity dict maintaining only the entities found in the corpus
    print('make found entity dict')
    found_entities = set(occurrences_of_entities.keys())
    found_entity_dict = {k: set(v).intersection(found_entities) for k,v in entity_dict.items() if set(v).intersection(found_entities)}
    
    if avoid_multilabeling:
        found_entity_dict = c.avoid_multilabeling(found_entity_dict, G, file = 'logs/avoid_multilabeling.txt')

    save_data_with_pickle(PICKLES_PATH + 'found_entity_dict_9_3', found_entity_dict)            
