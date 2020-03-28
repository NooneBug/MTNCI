
from utils import load_data_with_pickle, save_data_with_pickle
from ConceptEmbeddingManager import HyperEEmbeddingManager, Type2VecEmbeddingManager

# PATH in which utility files are stored
SOURCE_PATH = '../../source_files/'

# SOURCE_PATH = '/datahdd/vmanuel/MTNCI_datasets/source_files/'

PICKLES_PATH = SOURCE_PATH + 'pickles/'

# FILE_ID = '18_3_max2words_100000'
FILE_ID = '16_3_multilabel_'

FINAL_TREE_PATH = PICKLES_PATH + FILE_ID + 'final_tree'

TYPE_CORPUS = '/datahdd/vmanuel/MTNCI_datasets/source_files/type2vec_utilities/annotated_text_with_types.txt'

if __name__ == "__main__":
    
    concept_embedding_manager = HyperEEmbeddingManager()
    concept_embedding_manager.read_tree(tree_path = FINAL_TREE_PATH)
    concept_embedding_manager.create_embedding(embedding_name = FILE_ID + 'final_tree_HyperE_MTNCI_10',
                                                save_path = SOURCE_PATH + 'embeddings/',
                                                dimensions=10)

    concept_embedding_manager = Type2VecEmbeddingManager()
    # concept_embedding_manager.read_trees(PICKLES_PATH + FILE_ID + 'graph', FINAL_TREE_PATH) 
    concept_embedding_manager.read_trees(PICKLES_PATH + '18_3_max2words_100000graph', FINAL_TREE_PATH) 
    concept_embedding_manager.create_embedding(embedding_name = FILE_ID + 'final_tree_type2vec_MTNCI', 
                                                save_path = SOURCE_PATH + 'embeddings/',
                                               remove_mode = 'Remove', 
                                               concept_corpus_path = TYPE_CORPUS) 