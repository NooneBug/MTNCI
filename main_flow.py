
# %%
from MTNCI import MTNCI
import pickle
import torch
import sys
sys.path.append('./preprocessing/')
from utils import load_data_with_pickle, save_data_with_pickle
import numpy as np
from DatasetManager import DatasetManager

EMBEDDING_PATH = '../source_files/embeddings/'

PATH_TO_HYPERBOLIC_EMBEDDING = EMBEDDING_PATH + '10_3_final_tree_HyperE_MTNCI'

PATH_TO_DISTRIBUTIONAL_EMBEDDING = EMBEDDING_PATH + '10_3_final_tree_type2vec_MTNCI'

CONCEPT_EMBEDDING_PATHS = [PATH_TO_DISTRIBUTIONAL_EMBEDDING, PATH_TO_HYPERBOLIC_EMBEDDING]

DATASET_PATH = '../source_files/vectors/'

X_PATH = DATASET_PATH + 'X_10_3'
Y_PATH = DATASET_PATH + 'Y_10_3'
ENTITIES_PATH = DATASET_PATH + 'entities_10_3'

# %%
if __name__ == "__main__":
    
    # %%
    # if torch.cuda.is_available():
    #     device = torch.device("cuda")
    #     torch.set_default_tensor_type(torch.DoubleTensor)

    torch.cuda.manual_seed_all(236451)
    torch.manual_seed(236451)
    np.random.seed(236451)

    X = load_data_with_pickle(X_PATH)
    Y = load_data_with_pickle(Y_PATH)
    entities = load_data_with_pickle(ENTITIES_PATH)


    concept_embeddings = [load_data_with_pickle(x) for x in CONCEPT_EMBEDDING_PATHS]
    concept_embeddings = {'hyperbolic': concept_embeddings[1], 
                          'distributional':concept_embeddings[0]}
    
    dataset = DatasetManager()

    dataset.split_data_by_unique_entities(X = X, Y = Y, entities = entities)
    

# %%
