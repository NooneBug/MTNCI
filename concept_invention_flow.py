from DatasetManager import DatasetManager
from ConceptInventionManager import ConceptInventionManager

nickel = True


FILE_ID = 16_3

SOURCE_FILES_PATH = '/datahdd/vmanuel/MTNCI_datasets/source_files/'
# SOURCE_FILES_PATH = '../source_files/'

EMBEDDING_PATH = SOURCE_FILES_PATH + 'embeddings/'

PATH_TO_HYPERBOLIC_EMBEDDING = EMBEDDING_PATH + FILE_ID + '16_3_nickel.pth'

PATH_TO_DISTRIBUTIONAL_EMBEDDING = EMBEDDING_PATH + FILE_ID + 'final_tree_type2vec_MTNCI'

CONCEPT_EMBEDDING_PATHS = [PATH_TO_DISTRIBUTIONAL_EMBEDDING, 
                           PATH_TO_HYPERBOLIC_EMBEDDING]

DATA_PATH = '../../source_files/vectors/'
X_PATH = DATA_PATH + FILE_ID + 'X'
Y_PATH = DATA_PATH + FILE_ID + 'Y'
entities_PATH = DATA_PATH + FILE_ID + 'entities'

if __name__ == "__main__":
    

    datasetManager = DatasetManager(FILE_ID)
    datasetManager.load_entities_data(X_PATH = X_PATH,
                                        Y_PATH = Y_PATH,
                                        ENTITIES_PATH = ENTITIES_PATH)
    datasetManager.set_device(device)
    datasetManager.load_concept_embeddings(CONCEPT_EMBEDDING_PATHS = CONCEPT_EMBEDDING_PATHS, nickel = nickel)

    conceptInventionManager = ConceptInventionManager()
    conceptInventionManager.set_dataset_manager(datasetManager)

    list_of_concepts = set(datasetManager.Y)

    

    datasetManager.load_raw_dataset(FILTERED_DATASET_PATH + '0.6_3_aprile/')


