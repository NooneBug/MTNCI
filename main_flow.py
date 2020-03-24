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


SOURCE_FILES_PATH = '/datahdd/vmanuel/MTNCI_datasets/source_files/'
# SOURCE_FILES_PATH = '../source_files/'

EMBEDDING_PATH = SOURCE_FILES_PATH + 'embeddings/'

FILE_ID = '16_3'

PATH_TO_HYPERBOLIC_EMBEDDING = EMBEDDING_PATH + FILE_ID + 'final_tree_HyperE_MTNCI'

PATH_TO_DISTRIBUTIONAL_EMBEDDING = EMBEDDING_PATH + FILE_ID + 'final_tree_type2vec_MTNCI'

CONCEPT_EMBEDDING_PATHS = [PATH_TO_DISTRIBUTIONAL_EMBEDDING, 
                           PATH_TO_HYPERBOLIC_EMBEDDING]

DATASET_PATH = SOURCE_FILES_PATH + 'vectors/'

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
    threshold = 0.6

    if filter_dataset:
        filter = Filter()
        datasetManager.setup_filter(filter_name = filter.FILTERS['ClassCohesion'], 
                                    selfXY = True, 
                                    log_file_path = '../source_files/logs/19_3filter_log3', 
                                    filtered_dataset_path = SOURCE_FILES_PATH + 'vectors/{}filtered/'.format(FILE_ID),
                                    threshold=threshold)
        datasetManager.filter()
    
    fraction = 1

    datasetManager.shuffle_dataset_and_sample(fraction = fraction, in_place = True)

    datasetManager.split_data_by_unique_entities(exclude_min_threshold=10)
    print('Train: {} vectors, Val: {} vectors, Test: {} vectors'.format(len(datasetManager.Y_train),
                                                                        len(datasetManager.Y_val),
                                                                        len(datasetManager.Y_test)
                                                                        )
             )

    # datasetManager.plot_datasets()

    PICKLE_PATH = SOURCE_FILES_PATH + 'datasets/'
    ID = '1_16_3_filtered_{}'.format(threshold)
    # print('... saving dataset ...')
    # datasetManager.save_datasets(save_path = PICKLE_PATH, ID = ID)
    # datasetManager.print_statistic_on_dataset()
    print('... creating numeric dataset ...')
    datasetManager.create_numeric_dataset()
    print('... creating aligned dataset ...')
    datasetManager.create_aligned_dataset()
    print('... creating dataloaders ...')
    datasetManager.create_dataloaders()


    out_spec = [{'manifold':'euclid', 'dim':[64, len(datasetManager.aligned_y_train['distributional'][0])]},
                {'manifold':'poincare', 'dim':[128, 128, len(datasetManager.aligned_y_train['hyperbolic'][0])]}]

    model = MTNCI(input_d=len(datasetManager.X_train[0]),
                out_spec = out_spec,
                dims = [512, 512])

    model.set_dataset_manager(datasetManager)
    
    model.initialize_tensorboard_manager(ID)

    model.set_device(device)
    lr = 1e-3
    
    model.set_optimizer(optimizer = RiemannianAdam(model.parameters(), lr = lr))

    llambda = 0.1
    weighted = True
    epochs = 100

    model.set_lambda(llambdas = {'hyperbolic' : 1 - llambda,
                                 'distributional': llambda})
    


    model.set_hyperparameters(epochs = epochs, weighted=weighted)
    print('... training model ... ')
    model.train_model()

    topn = [1, 2, 5]

    for t in topn:
        model.type_prediction_on_test(topn=t)

    