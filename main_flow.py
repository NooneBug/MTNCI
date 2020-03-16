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

    distributional_prediction_manager = Prediction()
    distributional_prediction_manager.select_loss(distributional_prediction_manager.LOSSES['cosine_dissimilarity'])

    hyperbolic_prediction_manager = Prediction()
    hyperbolic_prediction_manager.select_loss(distributional_prediction_manager.LOSSES['hyperbolic_distance'])

    epochs = 20

    model.set_hyperparameters(epochs = epochs)


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
    # epochs = 2

    # def get_weight(l1, l2):
    #     return 2

    # for epoch in range(epochs):
    #     train_it = iter(trainloader)
    #     val_it = iter(valloader)
        
    #     train_loss_SUM = 0
    #     val_loss_SUM = 0
        
        
    #     distributional_train_loss_SUM = 0
    #     distributional_val_loss_SUM = 0
        
    #     hyperbolic_train_loss_SUM = 0
    #     hyperbolic_val_loss_SUM = 0
        
    #     for batch_iteration in range(len(trainloader)):
    #         x, labels, targets = next(train_it)
            
    #         ######################
    #         ####### TRAIN ########
    #         ######################
            
            
    #         optimizer.zero_grad()
    #         model.train()
            
    #         output = model(x)
            
    #         output_dict = {'distributional' : output[0],
    #                     'hyperbolic' : output[1]}
            
    #         distributional_train_loss = cosine_loss(output_dict['distributional'], 
    #                                                 targets['distributional'])
            
    #         hyperbolic_train_loss, r = hyperbolic_loss(output_dict['hyperbolic'], 
    #                                                 targets['hyperbolic'], 
    #                                                 regul = regul)
            
    #         if weighted:
    #             labels_weights_train = 3
    #             weights = 3
    #             weights_train = get_weight(labels, labels_weights_train)
    #             weights_train_SUM += torch.sum(weights).item()
    #             train_loss = torch.sum((distributional_train_loss * llambda + hyperbolic_train_loss * (1 - llambda)) * weights)
    #             distributional_train_loss_SUM += torch.sum(distributional_train_loss * llambda * weights).item()
    #             hyperbolic_train_loss_SUM += torch.sum(hyperbolic_train_loss * (1 - llambda) * weights).item()

    #         else:
    #             train_loss = torch.sum(distributional_train_loss * llambda + hyperbolic_train_loss * (1 - llambda))
    #             distributional_train_loss_SUM += torch.sum(distributional_train_loss * llambda).item()
    #             hyperbolic_train_loss_SUM += torch.sum(hyperbolic_train_loss * (1 - llambda)).item()
            
    #         train_loss_SUM += train_loss.item()
    #         train_loss.backward()
    #         optimizer.step()
            
            
    #     else:

    #         ######################
    #         ######## VAL #########
    #         ######################
            
    #         with torch.no_grad():
    #             model.eval()   
                
    #             for batch_iteration in range(len(valloader)):
    #                 x, labels, targets = next(val_it)
        
    #                 output = model(x)
    #                 output_dict = {'distributional' : output[0],
    #                             'hyperbolic' : output[1]}
                    
    #                 distributional_val_loss = cosine_loss(output_dict['distributional'],
    #                                                     targets['distributional'])
            
    #                 hyperbolic_val_loss, r = hyperbolic_loss(output_dict['hyperbolic'], 
    #                                                         targets['hyperbolic'], 
    #                                                         regul = [0, 0, 1])
                    
                    
    #                 if weighted:
    #                     labels_weights_val = 3
    #                     weights = 3
    #                     weights_val = get_weight(labels, labels_weights_val)
    #                     weights_val_SUM += torch.sum(weights).item()   
    #                     val_loss = torch.sum((distributional_val_loss * llambda + hyperbolic_val_loss * (1 - llambda)) * weights)
    #                     distributional_val_loss_SUM += torch.sum(distributional_val_loss * llambda * weights).item()
    #                     hyperbolic_val_loss_SUM += torch.sum(hyperbolic_val_loss * (1 - llambda) * weights).item()

    #                 else:
    #                     val_loss = torch.sum(distributional_val_loss * llambda + hyperbolic_val_loss * (1 - llambda))
    #                     distributional_val_loss_SUM += torch.sum(distributional_val_loss * llambda).item()
    #                     hyperbolic_val_loss_SUM += torch.sum(hyperbolic_val_loss * (1 - llambda)).item()
    #                 val_loss_SUM += val_loss.item()    
            
    #     print('{:^15}'.format('epoch {:^3}/{:^3}'.format(epoch, epochs)))
    #     print('{:^15}'.format('Train loss: {:.4f}, Val loss: {:.4f}'.format(train_loss_SUM/len(c.X_train), 
    #                                                                         val_loss_SUM/len(c.X_val)
    #                                                                     )
    #                         )
    #         )
            