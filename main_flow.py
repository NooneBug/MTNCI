# %%
from MTNCI import MTNCI, Prediction
import sys
import torch
import numpy as np
from DatasetManager import DatasetManager, Filter
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

    datasetManager = DatasetManager(FILE_ID)
    datasetManager.set_device(device)
    # load dataset
    datasetManager.load_entities_data(X_PATH = X_PATH,
                                      Y_PATH = Y_PATH,
                                      ENTITIES_PATH = ENTITIES_PATH)

    datasetManager.load_concept_embeddings(CONCEPT_EMBEDDING_PATHS = CONCEPT_EMBEDDING_PATHS)
    
    datasetManager.find_and_filter_not_present()

    normalize = True

    if normalize:
        datasetManager.normalize()

    filter_dataset = True

    if filter_dataset:
        filter = Filter()
        datasetManager.setup_filter(filter_name = filter.FILTERS['ClassCohesion'], 
                                    selfXY = True, 
                                    log_file_path = '../source_files/logs/filter_log', 
                                    filtered_dataset_path = '../source_files/vectors/{}filtered/'.format(FILE_ID))
        datasetManager.filter()
    
    # fraction = 0.5

    # datasetManager.shuffle_dataset_and_sample(fraction = fraction, in_place = True)

    # datasetManager.split_data_by_unique_entities(exclude_min_threshold=100)
    # print('Train: {} vectors, Val: {} vectors, Test: {} vectors'.format(len(datasetManager.Y_train),
    #                                                                     len(datasetManager.Y_val),
    #                                                                     len(datasetManager.Y_test)
    #                                                                     )
    #          )

    # # datasetManager.plot_datasets()

    # PICKLE_PATH = '../source_files/datasets/'
    # ID = '18_3_100_{}'.format(fraction)
    # print('... saving dataset ...')
    # datasetManager.save_datasets(save_path = PICKLE_PATH, ID = ID)
    # # datasetManager.print_statistic_on_dataset()
    # print('... creating numeric dataset ...')
    # datasetManager.create_numeric_dataset()
    # print('... creating aligned dataset ...')
    # datasetManager.create_aligned_dataset()
    # print('... creating dataloaders ...')
    # datasetManager.create_dataloaders()


    # out_spec = [{'manifold':'euclid', 'dim':[64, len(datasetManager.aligned_y_train['distributional'][0])]},
    #             {'manifold':'poincare', 'dim':[128, 128, len(datasetManager.aligned_y_train['hyperbolic'][0])]}]

    # model = MTNCI(input_d=len(datasetManager.X_train[0]),
    #             out_spec = out_spec,
    #             dims = [512, 512])

    # model.set_dataset_manager(datasetManager)
    
    # model.initialize_tensorboard_manager(ID)

    # model.set_device(device)
    # lr = 1e-3
    
    # model.set_optimizer(optimizer = RiemannianAdam(model.parameters(), lr = lr))

    # llambda = 0.1
    # weighted = True
    # epochs = 100

    # model.set_lambda(llambdas = {'hyperbolic' : 1 - llambda,
    #                              'distributional': llambda})
    


    # model.set_hyperparameters(epochs = epochs, weighted=weighted)
    # print('... training model ... ')
    # model.train_model()


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