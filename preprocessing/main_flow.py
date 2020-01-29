# %%
from EntityNameRetriever import EntityNameRetriever
from graph import graph_from_edgelist, remove_void_types
import networkx as nx
from CorpusManager import CorpusManager
import pickle
from utils import save_data_with_pickle, load_data_with_pickle

list_of_classes = ['dbo:City', 'dbo:Mosque', 'dbo:Animal']

PATH_TO_EDGELIST = './source_files/dbpedia_edgelist_no_closure.tsv'
CORPUS_PATH = '/datahdd/vmanuel/ELMo/Corpora/shuffled_text_with_words'

# %%
if __name__ == "__main__":

    # %%

    # create the ontology tree
    try:
        G = load_data_with_pickle('./pickles/graph')
    except:
        G = graph_from_edgelist(PATH_TO_EDGELIST)
        save_data_with_pickle('./pickles/graph', G)

    print("the input graph is a tree: {}".format(nx.is_tree(G)))
    # # %%
    
    list_of_classes = [n for n in G.nodes()]

    # retrieve entities names
    e = EntityNameRetriever()

    try:
        entity_dict = load_data_with_pickle('./pickles/entity_dict')
    except:
        entity_dict = e.entities_from_types(list_of_classes)
        entity_dict = e.entity_name_preprocessing(entity_dict)
        save_data_with_pickle('./pickles/entity_dict', entity_dict)

    save_data_with_pickle('./pickles/entity_dict', entity_dict)
    
    # exclude from the tree the concepts which have no entities (only if the concept is a leaf or have all empty sons)
    void_types = [t for t, v in entity_dict.items() if v == []]

    # # %%
    pruned_G = remove_void_types(G, void_types)
    print("the pruned graph is a tree: {}".format(nx.is_tree(pruned_G)))


    c = CorpusManager()
    try:
        1/0
        c.vocab = load_data_with_pickle('./pickles/vocab')
        c.concept_entities = load_data_with_pickle('./pickles/concept_entities')
        c.corpus = load_data_with_pickle('../../MTNCI_github/preprocessing/pickles/corpus')

    except:
        c.read_corpus(CORPUS_PATH)
        c.check_words(entity_dict)
        save_data_with_pickle('./pickles/corpus', c.corpus)
        save_data_with_pickle('./pickles/vocab', c.vocab)
        save_data_with_pickle('./pickles/concept_entities', c.concept_entities)

    
    print(list(c.concept_entities.keys()))


# %%
