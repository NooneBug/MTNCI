
from utils import load_data_with_pickle, save_data_with_pickle
from ConceptEmbeddingManager import HyperEEmbeddingManager, Type2VecEmbeddingManager

# PATH in which utility files are stored
PICKLES_PATH = '../../source_files/pickles/'

FILE_ID = '16_3'

FINAL_TREE_PATH = PICKLES_PATH + FILE_ID + 'final_tree'

TYPE_CORPUS = '../../source_files/type2vec_utilities/annotated_text_with_types.txt'

if __name__ == "__main__":
    
    concept_embedding_manager = HyperEEmbeddingManager()
    concept_embedding_manager.read_tree(tree_path = FINAL_TREE_PATH)
    concept_embedding_manager.create_embedding(FILE_ID + 'final_tree_HyperE_MTNCI')

    concept_embedding_manager = Type2VecEmbeddingManager()
    concept_embedding_manager.read_trees(PICKLES_PATH + FILE_ID + 'graph', FINAL_TREE_PATH) 
    concept_embedding_manager.create_embedding(embedding_name = FILE_ID + 'final_tree_type2vec_MTNCI', 
                                               remove_mode = 'Remove', 
                                               concept_corpus_path = TYPE_CORPUS) 