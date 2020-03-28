# %%
from MTNCI import MTNCI, Prediction
import sys
import torch
import numpy as np
from DatasetManager import DatasetManager, Filter
from preprocessing.utils import load_data_with_pickle, save_data_with_pickle, euclidean_similarity
from preprocessing.CorpusManager import CorpusManager
from geoopt.optim import RiemannianAdam
import time
from sklearn.metrics.pairwise import cosine_similarity

multilabel = False
load_dataset = False
filter_dataset = True
threshold = 0.5
distance = euclidean_similarity
normalize = True
tensorboard_run_ID = '1_strange_{}_nickel'.format(threshold)
# tensorboard_run_ID = '1_16_3_multilabel_filtered_{}'.format(threshold)
nickel = True
FILE_ID = '16_3'


SOURCE_FILES_PATH = '/datahdd/vmanuel/MTNCI_datasets/source_files/'
# SOURCE_FILES_PATH = '../source_files/'

EMBEDDING_PATH = SOURCE_FILES_PATH + 'embeddings/'

# PATH_TO_HYPERBOLIC_EMBEDDING = EMBEDDING_PATH + FILE_ID + 'final_tree_HyperE_MTNCI'
# PATH_TO_HYPERBOLIC_EMBEDDING = EMBEDDING_PATH + FILE_ID + 'final_tree_HyperE_MTNCI_32'
# PATH_TO_HYPERBOLIC_EMBEDDING = EMBEDDING_PATH + FILE_ID + '_multilabel_final_tree_HyperE_MTNCI_10'
PATH_TO_HYPERBOLIC_EMBEDDING = EMBEDDING_PATH + FILE_ID + '16_3_nickel.pth'


PATH_TO_DISTRIBUTIONAL_EMBEDDING = EMBEDDING_PATH + FILE_ID + 'final_tree_type2vec_MTNCI'

CONCEPT_EMBEDDING_PATHS = [PATH_TO_DISTRIBUTIONAL_EMBEDDING, 
                           PATH_TO_HYPERBOLIC_EMBEDDING]

DATASET_PATH = SOURCE_FILES_PATH + 'vectors/'

    
if not multilabel:
    X_PATH = DATASET_PATH + FILE_ID + 'X'
    Y_PATH = DATASET_PATH + FILE_ID + 'Y'
    ENTITIES_PATH = DATASET_PATH + FILE_ID + 'entities'
else:
    X_PATH = DATASET_PATH + FILE_ID + 'X_multilabel'
    Y_PATH = DATASET_PATH + FILE_ID + 'Y_multilabel'
    ENTITIES_PATH = DATASET_PATH + FILE_ID + 'entities_multilabel'
    

FILTERED_DATASET_PATH = '../source_files/vectors/' + FILE_ID + '/' 

X_TRAIN_PATH = FILTERED_DATASET_PATH + 'filtered_X_train'
X_VAL_PATH = FILTERED_DATASET_PATH + 'filtered_X_val' 
X_TEST_PATH = FILTERED_DATASET_PATH + 'filtered_X_test'
Y_TRAIN_PATH = FILTERED_DATASET_PATH + 'filtered_Y_train'
Y_VAL_PATH = FILTERED_DATASET_PATH + 'filtered_Y_val' 
Y_TEST_PATH = FILTERED_DATASET_PATH + 'filtered_Y_test'
E_TRAIN_PATH = FILTERED_DATASET_PATH + 'filtered_entities_train'
E_VAL_PATH = FILTERED_DATASET_PATH + 'filtered_entities_val'
E_TEST_PATH = FILTERED_DATASET_PATH + 'filtered_entities_test'

# %%
if __name__ == "__main__":
    
    # %%
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.set_default_tensor_type(torch.DoubleTensor)
    
    datasetManager = DatasetManager(FILE_ID)
    datasetManager.set_device(device)

    datasetManager.load_concept_embeddings(CONCEPT_EMBEDDING_PATHS = CONCEPT_EMBEDDING_PATHS, nickel = nickel)

    if not load_dataset:
    # create dataset
        datasetManager.load_entities_data(X_PATH = X_PATH,
                                        Y_PATH = Y_PATH,
                                        ENTITIES_PATH = ENTITIES_PATH)

        
        datasetManager.find_and_filter_not_present()


        if normalize:
            datasetManager.normalize()

        if filter_dataset:
            filter = Filter()
            datasetManager.setup_filter(filter_name = filter.FILTERS['ClassCohesion'], 
                                        selfXY = True, 
                                        log_file_path = '../source_files/logs/19_3filter_log3', 
                                        filtered_dataset_path = SOURCE_FILES_PATH + 'vectors/{}filtered/'.format(FILE_ID),
                                        threshold=threshold,
                                        cluster_distance = distance)
            datasetManager.filter()
        
        fraction = 1

        datasetManager.shuffle_dataset_and_sample(fraction = fraction, in_place = True)

        datasetManager.split_data_by_unique_entities(exclude_min_threshold=10)
        print('Train: {} vectors, Val: {} vectors, Test: {} vectors'.format(len(datasetManager.Y_train),
                                                                            len(datasetManager.Y_val),
                                                                            len(datasetManager.Y_test)
                                                                            )
                )


        # print('... saving dataset ...')
        # datasetManager.save_datasets(save_path = FILTERED_DATASET_PATH)

    else:
        print('... loading datasets ...')
        t = time.time()
        datasetManager.X_train = load_data_with_pickle(X_TRAIN_PATH)
        datasetManager.X_test = load_data_with_pickle(X_TEST_PATH)
        datasetManager.X_val = load_data_with_pickle(X_VAL_PATH)
        datasetManager.Y_train = load_data_with_pickle(Y_TRAIN_PATH)
        datasetManager.Y_test = load_data_with_pickle(Y_TEST_PATH)
        datasetManager.Y_val = load_data_with_pickle(Y_VAL_PATH)
        datasetManager.E_train = load_data_with_pickle(E_TRAIN_PATH)
        datasetManager.E_test = load_data_with_pickle(E_TEST_PATH)
        datasetManager.E_val = load_data_with_pickle(E_VAL_PATH)
        print('--- dataset loaded in {:.2f} seconds ---'.format(time.time() - t))

    # datasetManager.plot_datasets()
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

    model.set_checkpoint_path(checkpoint_path = '../source_files/checkpoints/{}'.format(tensorboard_run_ID))

    model.set_dataset_manager(datasetManager)
    
    model.initialize_tensorboard_manager(tensorboard_run_ID)

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
    model.type_prediction_on_test(topn=topn)

    