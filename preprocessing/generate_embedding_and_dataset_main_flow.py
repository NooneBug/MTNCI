import pickle
from utils import save_data_with_pickle, load_data_with_pickle
from EntityEmbedder import EntityEmbedder
from CorpusManager import CorpusManager


# PATH in which utility files are stored
PICKLES_PATH = '../../source_files/pickles/'
OCCURRENCE_OF_ENTITIES_PATH = PICKLES_PATH + 'occurrences_of_entities_9_3'
ENTITY_DICT_PATH = PICKLES_PATH + 'found_entity_dict_9_3'

CORPUS_PATH = '/datahdd/vmanuel/ELMo/Corpora/shuffled_text_with_words'
LENGTH = 100000

DATA_PATH = '../../source_files/vectors/'
X_PATH = DATA_PATH + 'X_9_3'
Y_PATH = DATA_PATH + 'Y_9_3'
entities_PATH = DATA_PATH + 'entities_9_3'

if __name__ == "__main__":
    entity_dict = load_data_with_pickle(ENTITY_DICT_PATH)

    c = CorpusManager()
    c.read_corpus(CORPUS_PATH, LENGTH)

    entity_embedder = EntityEmbedder()
    entity_embedder.setup(model_name = entity_embedder.ELMO_NAME,
                          extraction_mode = entity_embedder.LAYER_2,
                          occurrences_of_entities_path = OCCURRENCE_OF_ENTITIES_PATH,
                          aggregation_method = entity_embedder.VECTOR_MEAN,
                          corpus = c.corpus
                          )

    entity_embedder.create_embedding_data_structure()
    entity_embedder.extract_vectors_of_occurrences_in_corpus()
    entity_embedder.create_dataset(entity_dict = entity_dict, 
                                   X_PATH = X_PATH,
                                   Y_PATH = Y_PATH,
                                   entities_PATH = entities_PATH)


        