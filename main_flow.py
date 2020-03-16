# %%
from MTNCI import MTNCI, Prediction
import sys
import torch
import numpy as np
from DatasetManager import DatasetManager
sys.path.append('./preprocessing/')
from utils import load_data_with_pickle, save_data_with_pickle
from CorpusManager import CorpusManager
from geoopt.optim import RiemannianAdam



EMBEDDING_PATH = '../source_files/embeddings/'

FILE_ID = '16_3'

PATH_TO_HYPERBOLIC_EMBEDDING = EMBEDDING_PATH + FILE_ID + 'final_tree_HyperE_MTNCI'

PATH_TO_DISTRIBUTIONAL_EMBEDDING = EMBEDDING_PATH + FILE_ID + 'final_tree_type2vec_MTNCI'

CONCEPT_EMBEDDING_PATHS = [PATH_TO_DISTRIBUTIONAL_EMBEDDING, 
                           PATH_TO_HYPERBOLIC_EMBEDDING]

DATASET_PATH = '../source_files/vectors/'

X_PATH = DATASET_PATH + FILE_ID + 'X'
Y_PATH = DATASET_PATH + FILE_ID + 'Y'
ENTITIES_PATH = DATASET_PATH + FILE_ID + 'entities'

# %%
if __name__ == "__main__":
    
    # %%
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.set_default_tensor_type(torch.DoubleTensor)

    torch.manual_seed(236451)
    np.random.seed(236451)

    datasetManager = DatasetManager(FILE_ID)
    # load dataset
    datasetManager.load_entities_data(X_PATH = X_PATH,
                                      Y_PATH = Y_PATH,
                                      ENTITIES_PATH = ENTITIES_PATH)

    datasetManager.load_concept_embeddings(CONCEPT_EMBEDDING_PATHS = CONCEPT_EMBEDDING_PATHS)
    
    datasetManager.find_and_filter_not_present()

    datasetManager.shuffle_dataset_and_sample(fraction = 0.1, in_place = True)

    datasetManager.split_data_by_unique_entities(exclude_min_threshold=100)
    
    PICKLE_PATH = '../source_files/datasets/'
    ID = '16_3_100_0.1'

    datasetManager.save_datasets(save_path = PICKLE_PATH, ID = ID)
    # datasetManager.print_statistic_on_dataset()

    datasetManager.create_numeric_datasets()

    datasetManager.create_aligned_dataset()

    datasetManager.create_dataloaders()


    out_spec = [{'manifold':'euclid', 'dim':[64, len(explicit_y_train['distributional'][0])]},
                {'manifold':'poincare', 'dim':[128, len(explicit_y_train['hyperbolic'][0])]}]

    model = MTNCI(input_d=len(datasetManager.X_train[0]),
                out_spec = out_spec,
                dims = [256])

    lr = 1e-3
    
    model.set_optimizer(optimizer = RiemannianAdam(model.parameters(), lr = lr))

    llambda = 0.05

    model.set_lambda(llambdas = {'hyperbolic' : 1 - llambda,
                                 'distributional': llambda)

    epochs = 20

    model.set_hyperparameters(epochs = epochs)

    model.train()

    # weighted = False

    # min_loss = [100, 2000]
    # min_val_loss = [100, 2000]
    # best_sim = [-1, 4000]
    # sims = [0, 0]
    # sim = [0, 0]
    # colored, val_colored = [], []
    # checkpoint_path = './models/MTN'
    # epochs_no_improve = 0
    # n_epochs_stop = 20
    # min_val_losses = 100

    # def get_weight(l1, l2):
    #     return 2

    #       