
from utils import load_data_with_pickle, save_data_with_pickle
from ConceptEmbeddingManager import HyperEEmbeddingManager, Type2VecEmbeddingManager

# PATH in which utility files are stored
PICKLES_PATH = '../../source_files/pickles/'

FILE_ID = '10_3_'

FINAL_TREE_PATH = PICKLES_PATH + FILE_ID + 'final_tree'

TYPE_CORPUS = '../../source_files/type2vec_utilities/annotated_text_with_types.txt'

if __name__ == "__main__":
    
    concept_embedding_manager = Type2VecEmbeddingManager(concept_corpus_path = TYPE_CORPUS)

    concept_embedding_manager.read_tree(PICKLES_PATH + '10_3_graph') 
    concept_embedding_manager.create_embedding(embedding_name = 'MTNCI_type2vec', remove_mode = 'Remove') 

    
