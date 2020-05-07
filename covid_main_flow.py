from MTNCI.MTNCI import ShimaokaMTNCI
# from MTNCI import LopezLike as ShimaokaMTNCI
import torch
from MTNCI.DatasetManager import ShimaokaMTNCIDatasetManager as DatasetManager
import sys 
sys.path.append('../figet-hyperbolic-space')
import figet
from geoopt.optim import RiemannianAdam
from tqdm import tqdm
from preprocessing.utils import LOSSES


import time
import telegram

def send(msg, chat_id = 190649040, token='792681420:AAH_wiAsCO5Bk1kan_Iy3LTaJjDl3gWOZBU'):
  bot = telegram.Bot(token=token)
  bot.sendMessage(chat_id=chat_id, text=msg)


class argClass():
    
    def __init__(self, args):
        self.emb_size = 300 
        self.char_emb_size = 50 
        self.positional_emb_size = 25 
        self.context_rnn_size = 200
        self.attn_size = 100
        self.mention_dropout = 0.5
        self.context_dropout = 0.5

# lopez_data = torch.load('../figet-hyperbolic-space/data/prep/lopez_covid/data.pt')
# word2vec = torch.load('../figet-hyperbolic-space/data/prep/lopez_covid/word2vec.pt')

lopez_data = torch.load('../figet-hyperbolic-space/data/prep/single_ontonotes/data.pt')
word2vec = torch.load('../figet-hyperbolic-space/data/prep/single_ontonotes/word2vec.pt')



losses_dict = {
    'hyperbolic-train': LOSSES['hyperbolic_distance'],
    'hyperbolic-val': LOSSES['hyperbolic_distance'],
    'distributional': LOSSES['cosine_dissimilarity']
}

metrics_dict = {
    'distributional': LOSSES['cosine_dissimilarity'],
    'hyperbolic': LOSSES['hyperbolic_distance']
}


args = {'emb_size': 300, 'char_emb_size': 50, 'positional_emb_size': 25, 'context_rnn_size':200,
        'attn_size': 100, 'mention_dropout' : 0.5, 'context_dropout': 0.2}
args = argClass(args)
vocabs = lopez_data['vocabs']
dims = [512, 512]
SHIMAOKA_OUT = args.context_rnn_size * 2 + args.emb_size + args.char_emb_size
out_spec = [{'manifold':'euclid', 'dim':[256, 10]},
            # {'manifold':'poincare', 'dim':[128, 128, 10]}]
            {'manifold':'poincare', 'dim':[512, 256, 10]}]

NAME = 'ontonotes_weighted'

regularized = False
regul_dict = {'negative_sampling': 0, 'mse': 50, 'distance_power':1}


tensorboard_run_ID = NAME
results_path = 'results/ontonotes_single/' + NAME + '.txt'
TSV_path = 'results/ontonotes_single_export/export_' + NAME + '.txt'

lr = 1e-3

llambda = 0.5
weighted = True
epochs = 50
patience = 50

times = 8
perc = 0.1

nickel = True
cleaned = True
# tensorboard_run_ID = '1_shimaoka_hinge_nickel'

FILE_ID = '16_3'

SOURCE_FILES_PATH = '/datahdd/vmanuel/MTNCI_datasets/source_files/'
# SOURCE_FILES_PATH = '../source_files/'

# EMBEDDING_PATH = './covid/models/'
# PATH_TO_HYPERBOLIC_EMBEDDING = EMBEDDING_PATH + 'covid.pth.best'
# PATH_TO_DISTRIBUTIONAL_EMBEDDING = EMBEDDING_PATH + 'type2vec_cleaned'


EMBEDDING_PATH = '../figet-hyperbolic-space/data/type-embeds/'
PATH_TO_HYPERBOLIC_EMBEDDING = EMBEDDING_PATH + 'ontonotes-clean.pth'
PATH_TO_DISTRIBUTIONAL_EMBEDDING = EMBEDDING_PATH + 'ontonotes_type2vec_cleaned'



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
    datasetManager.load_concept_embeddings(CONCEPT_EMBEDDING_PATHS = CONCEPT_EMBEDDING_PATHS, nickel = nickel, cleaned = cleaned)
    
    train_data = {'data': train, 'labels': train_labels}
    val_data = {'data': val, 'labels': val_labels}
    test_data = {'data': test, 'labels': test_labels}
    print('setup datamanager')
    datasetManager.set_batched_data(train_data, val_data, test_data)
    
    t = time.time()
    for i in range(1, 5):
        model = ShimaokaMTNCI(args, vocabs, device, 
                        input_d=SHIMAOKA_OUT,
                        out_spec = out_spec,
                        dims = dims)
        
        model.init_params(word2vec=word2vec)

        model.set_dataset_manager(datasetManager)
        
        model.set_checkpoint_path(checkpoint_path = '../source_files/checkpoints/{}'.format(tensorboard_run_ID))

        model.initialize_tensorboard_manager(tensorboard_run_ID)

        model.set_device(device)
        
        model.set_optimizer(optimizer = RiemannianAdam(model.parameters(), lr = lr))

        model.set_losses(losses_dict)
        model.set_metrics(metrics_dict)


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

        topn = [1, 3, 5]

        model.type_prediction_on_test(topn, test_data, test_entities, test_labels)

        send('{}: model {} done in {} seconds!'.format(NAME, i, int(time.time() - t)))
        t = time.time()

    # model.type_prediction_on_test(topn, train_data, train_entities, train_labels)