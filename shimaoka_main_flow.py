from MTNCI import ShimaokaMTNCI
# from MTNCI import LopezLike as ShimaokaMTNCI
import torch
from DatasetManager import ShimaokaMTNCIDatasetManager as DatasetManager
import sys 
sys.path.append('../figet-hyperbolic-space')
import figet
from geoopt.optim import RiemannianAdam
from tqdm import tqdm

from AAA import send

class argClass():
    
    def __init__(self, args):
        self.emb_size = 300 
        self.char_emb_size = 50 
        self.positional_emb_size = 25 
        self.context_rnn_size = 200
        self.attn_size = 100
        self.mention_dropout = 0.5
        self.context_dropout = 0.5

lopez_data = torch.load('../figet-hyperbolic-space/data/prep/correct-20/data.pt')
word2vec = torch.load('../figet-hyperbolic-space/data/prep/correct-20/word2vec.pt')


args = {'emb_size': 300, 'char_emb_size': 50, 'positional_emb_size': 25, 'context_rnn_size':200,
        'attn_size': 100, 'mention_dropout' : 0.5, 'context_dropout': 0.2}
args = argClass(args)
vocabs = lopez_data['vocabs']
SHIMAOKA_OUT = args.context_rnn_size * 2 + args.emb_size + args.char_emb_size
out_spec = [{'manifold':'euclid', 'dim':[64, 10]},
            {'manifold':'poincare', 'dim':[128, 128, 10]}]

NAME = 'correct_5'

regularized = False
regul_dict = {'negative_sampling': 0, 'mse': 50, 'distance_power':1}


tensorboard_run_ID = NAME
results_path = 'results/excel_results/' + NAME + '.txt'
TSV_path = 'results/excel_results/export_' + NAME + '.txt'

llambda = 0.8
weighted = True
epochs = 50
patience = 50

times = 1
perc = 1

nickel = True
# tensorboard_run_ID = '1_shimaoka_hinge_nickel'

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


if __name__ == "__main__":

    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.set_default_tensor_type(torch.DoubleTensor)

    def get_dataset(data, batch_size, key):
        dataset = data[key]
        dataset.set_batch_size(batch_size)
        return dataset

    print('load dataset')
    train = get_dataset(lopez_data, 1024, "train")
    test = get_dataset(lopez_data, 1024, "test")
    val = get_dataset(lopez_data, 1024, "dev")
    print('transform dataset')
    
    train_labels = [lopez_data['vocabs']['type'].idx2label[label.item()] for entry in train for labels in entry[5] for label in labels]
    val_labels = [lopez_data['vocabs']['type'].idx2label[label.item()] for entry in val for labels in entry[5] for label in labels]
    test_labels = [lopez_data['vocabs']['type'].idx2label[label.item()] for entry in test for labels in entry[5] for label in labels]
    
    # train_entities = []
    # for entry in tqdm(train): 
    #     for entities in entry[3]:
    #         entity_label = ''
    #         for entity in entities:
    #             if entity.item() != 0:
    #                 entity_label += lopez_data['vocabs']['token'].idx2label[entity.item()] + ' '
    #         train_entities.append(entity_label)
    
    test_entities = []
    for entry in tqdm(test): 
        for entities in entry[3]:
            entity_label = ''
            for entity in entities:
                if entity.item() != 0:
                    entity_label += lopez_data['vocabs']['token'].idx2label[entity.item()] + ' '
            test_entities.append(entity_label)

    datasetManager = DatasetManager(FILE_ID)
    datasetManager.set_device(device)
    datasetManager.load_concept_embeddings(CONCEPT_EMBEDDING_PATHS = CONCEPT_EMBEDDING_PATHS, nickel = nickel)
    
    train_data = {'data': train, 'labels': train_labels}
    val_data = {'data': val, 'labels': val_labels}
    test_data = {'data': test, 'labels': test_labels}
    print('setup datamanager')
    datasetManager.set_batched_data(train_data, val_data, test_data)
    
    model = ShimaokaMTNCI(args, vocabs, device, 
                    input_d=SHIMAOKA_OUT,
                    out_spec = out_spec,
                    dims = [512, 512])
    
    model.init_params(word2vec=word2vec)

    model.set_dataset_manager(datasetManager)
    
    model.set_checkpoint_path(checkpoint_path = '../source_files/checkpoints/{}'.format(tensorboard_run_ID))

    model.initialize_tensorboard_manager(tensorboard_run_ID)

    model.set_device(device)
    lr = 1e-3
    
    model.set_optimizer(optimizer = RiemannianAdam(model.parameters(), lr = lr))

    

    model.set_lambda(llambdas = {'hyperbolic' : 1 - llambda,
                                 'distributional': llambda})
    

    model.set_results_paths(results_path = results_path, TSV_path = TSV_path)


    model.set_hyperparameters(epochs = epochs, 
                                weighted=weighted, 
                                regularized=regularized, 
                                patience = patience,
                                times = times,
                                perc = perc)
    
    if regularized:
        model.set_regularization_params(regul_dict)

    print('... training model ... ')
    model.train_model()

    topn = [1, 2, 5]

    model.type_prediction_on_test(topn, test_data, test_entities, test_labels)
    # model.type_prediction_on_test(topn, train_data, train_entities, train_labels)