# %%
from EntityNameRetriever import EntityNameRetriever
from graph import graph_from_edgelist, remove_void_types
import networkx as nx
from CorpusManager import CorpusManager
import pickle
from utils import save_data_with_pickle, load_data_with_pickle
import time

list_of_classes = ['dbo:City', 'dbo:Mosque', 'dbo:Animal']
PICKLES_PATH = '../../source_files/pickles/'

PATH_TO_EDGELIST = PICKLES_PATH + 'dbpedia_edgelist_no_closure.tsv'
CORPUS_PATH = '/datahdd/vmanuel/ELMo/Corpora/shuffled_text_with_words'


# %%
if __name__ == "__main__":

    # %%

    # create the ontology tree
    try:
        G = load_data_with_pickle(PICKLES_PATH + 'graph')
    except:
        G = graph_from_edgelist(PATH_TO_EDGELIST)
        save_data_with_pickle(PICKLES_PATH + 'graph', G)

    print("the input graph is a tree: {}".format(nx.is_tree(G)))
    # # %%
    
    list_of_classes = [n for n in G.nodes()]

    # retrieve entities names
    e = EntityNameRetriever()

    try:
        entity_dict = load_data_with_pickle(PICKLES_PATH + 'entity_dict')
    except:
        entity_dict = e.entities_from_types(list_of_classes)
        entity_dict = e.entity_name_preprocessing(entity_dict)
        save_data_with_pickle(PICKLES_PATH + 'entity_dict', entity_dict)

    save_data_with_pickle(PICKLES_PATH + 'entity_dict', entity_dict)
    
    # exclude from the tree the concepts which have no entities (only if the concept is a leaf or have all empty sons)
    void_types = [t for t, v in entity_dict.items() if v == []]

    # # %%
    pruned_G = remove_void_types(G, void_types)
    print("the pruned graph is a tree: {}".format(nx.is_tree(pruned_G)))


    c = CorpusManager()
    c.read_corpus(CORPUS_PATH)
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

    c.create_all_entities(entity_dict)
    
    processes = [5, 6, 7, 8]
    entities = [100, 200, 500]

    for p in processes:
        for e in entities:
            f = c.parallel_find(n_proc = 6, n_entities= e)
    # save_data_with_pickle(PICKLES_PATH + 'word_occurrence_index', c.word_occurrence_index)
    # save_data_with_pickle(PICKLES_PATH + 'word_indexes_to_rebuild', f)
    # c.create_word_to_index()
    # t = time.time()    
    # result, slices = c.parallel_find(5)
    # print(round(time.time() - t, 2))
    # c.find()
    # save_data_with_pickle(PICKLES_PATH + 'result', result)
    # save_data_with_pickle(PICKLES_PATH + 'slices', slices)


# %%
