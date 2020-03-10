from allennlp.commands.elmo import ElmoEmbedder
from collections import OrderedDict
from utils import save_data_with_pickle, load_data_with_pickle
from tqdm import tqdm
from collections import defaultdict
import numpy as np

class EntityEmbedder:

    def __init__(self):
        """
        Initialize the value for constants which are useful to drive the behaviour 
        """
        # list of model names
        self.ELMO_NAME = 'Elmo'

        # list of extraction modes for Elmo
        self.LAYER_2 = 'layer_2'
        self.LAYER_1 = 'layer_1'
        self.LAYER_0 = 'layer_0'
        self.MEAN = 'mean'

        # list of word phrase aggregation names
        self.VECTOR_MEAN = 'vector_mean'

    def initialize_embedder_model(self, model_name, corpus):
        """
        setup the variables which will determine how words will be translated in vectors
        :param 
            model_name: the string which identifies the model used to embed
                values: 
                    ELMO_NAME: the allennlp's ElmoEmbedder is used, this take one sentence at time
                               a sentence is a list of single words (['this', 'is', 'a', 'sentence']) 
            corpus: a corpus in a format in accord with the model specifications:
                ELMO_NAME:
                    a list of lists, each sublist is a sentence in the format ['this', 'is', 'a', 'sentence'] 
        """

        if model_name == self.ELMO_NAME:
            self.model = ElmoEmbedder(cuda_device = 0)
            self.model_name = model_name
            self.corpus = corpus

    def setup(self, model_name, extraction_mode, occurrences_of_entities_path, aggregation_method, corpus, verbose = False):
        """
        setup the values to drive the behaviour and setup the resources
        :param
            model_name: the name of the embedding model (ELMO_NAME)
            extraction_mode: the modality to extract vectors for word:
                if model_name == ELMO_NAME then extraction_mode can take these values:
                    [LAYER_0, LAYER_1, LAYER_2]: the vector returned comes from layer 0 / 1 / 2 of ELMO
                    MEAN: the mean of layers 0, 1 and 2 is returned
            occurrences_of_entities_path: the path to the file which contains the occurrences of the entities (the output of CorpusManager.check_composite_words())
            aggregation_method: the method used to aggregate token vectors in word phrases (for 'new york' there will be two vectors, we want only one)
                values:
                    VECTOR_MEAN: the mean of all token vectors is returned
            corpus: a corpus in a format in accord with the model specifications (see inizialize_embedder_model for more specific description)
        :return: a list of indexes which are the row in which word appear
        """
        print('setupping the embedder')
        self.initialize_embedder_model(model_name = model_name, corpus = corpus)
        self.extraction_mode = extraction_mode
        self.OCCURRENCE_OF_ENTITIES_PATH = occurrences_of_entities_path
        self.verbose = verbose
        self.aggregation_method = aggregation_method

    def set_extraction_mode(self, mode):
        """
        setup the name of the extraction mode, which will be used to drive the other functions in the class
        :param 
            mode: the string which identifies the mode used to extract vectors of the sentences 
        """
        self.extraction_mode = mode

    def extract_embedding(self, model_output):
        """
        returns the embeddings starting from the output of the model
        :param 
            model_output: the desired output of self.model 
        """

        if self.extraction_mode == self.LAYER_2 and self.model_name == self.ELMO_NAME:
            return model_output[2]
        if self.extraction_mode == self.LAYER_1 and self.model_name == self.ELMO_NAME:
            return model_output[1]
        if self.extraction_mode == self.LAYER_0 and self.model_name == self.ELMO_NAME:
            return model_output[0]
        if self.extraction_mode == self.MEAN and self.model_name == self.ELMO_NAME:
            return (model_output[0] + model_output[1] + model_output[2]) / 3
        

    def embed_sentence(self, sentence):
        """
        returns the embedding of the input sentence based on the instantiated model
        :param 
            sentence: if model_name == ELMO_NAME a sentence in this format: ['this', 'is', 'a', 'sentence']
        """
        if self.model_name == self.ELMO_NAME:
            return self.extract_embedding(self.model.embed_sentence(sentence))
    
    def create_embedding_data_structure(self):
        """
        creates the data structure useful to retrieve embeddings
        needs the output of the function 'check_composite_words' of the CorpusManager Class
        """

        print('creating data structures')
        all_occurrences = load_data_with_pickle(self.OCCURRENCE_OF_ENTITIES_PATH)
        all_occurrences = [(k , v) if type(v[0]) == tuple else (k, v[0]) for k, v in all_occurrences.items() if len(k) > 2]
        all_occurrences = {x[0]: x[1] for x in all_occurrences}

        sentences_to_embed = [v[0] for values in all_occurrences.values() for v in values]

        if self.verbose:
            print('total found entity mentions: {}'.format(len(sentences_to_embed)))
            print('fraction of sentences with entity mentions: {:.2f} ({} on {})'.format(len(set(sentences_to_embed))/len(self.corpus),
                                                                                        len(set(sentences_to_embed)),
                                                                                        len(self.corpus)))
            print('{:.2f} average entity mentions per sentence'.format(len(sentences_to_embed)/len(set(sentences_to_embed))))

        embedding_data_structure = {index:[] for index in sentences_to_embed}

        for entity_mention, occurrences in all_occurrences.items():
            for couple in occurrences:
                embedding_data_structure[couple[0]].append((couple[1], entity_mention))
        
        embedding_data_structure = {k:v for k,v in embedding_data_structure.items() if v }
        self.ordered_embedding_data_structure = OrderedDict(sorted(embedding_data_structure.items()))

    def extract_vectors_of_occurrences_in_corpus(self):
        """
        returns the embedding of all input sentences based on the instantiated model
        :param 
            sentences: if model_name == ELMO_NAME a list of sentence in this format: ['this', 'is', 'a', 'sentence']
        """
        print('generate vectors')
        self.vectors_dict = defaultdict(list)

        for row_index, occurrences in tqdm(self.ordered_embedding_data_structure.items()):
            vectors = self.embed_sentence(self.corpus[row_index])
            for occ in occurrences:
                for word_index in occ[0]:
                    if len(occ[1].split(' ')) == 1:
                        self.vectors_dict[occ[1]].append(vectors[word_index])
                    else:
                        vecs = [vectors[w_i] for w_i in range(word_index, word_index + len(occ[1].split(' ')))]
                        self.vectors_dict[occ[1]].append(self.word_phrase_aggregation_method(vecs = vecs))
    
    def word_phrase_aggregation_method(self, vecs):
        """
        aggregates a list of vectors in accord to the aggregation method 
        (extracts a single vector for the word phrase 'New York' starting from the vectors of 'New' and 'York')
        :param 
            vecs: the list of vector to be aggregated
        """
        if self.aggregation_method == self.VECTOR_MEAN :
            return np.mean(vecs, axis = 0)
        
    def create_dataset(self, entity_dict, X_PATH, Y_PATH, entities_PATH):
        """
        creates a dataset composed of: a list of vectors (X), a list of labels (Y), the entities names which order corresponds to values in X and Y (entities)
        :param 
            entity_dict: a dict of entities which is in the format: {concept: [list of entities]}, used to set the Y values and the entities values
            X_PATH: the filepath in which save the list of vectors
            Y_PATH: the filepath in which save the list of labels
            entities_PATH: the filepath in which save the list of entities names

        """
        print('creating dataset')

        reverse_dict = defaultdict(list)

        for k, words in entity_dict.items():
            for w in words:
                reverse_dict[w].append(k)
        X = []
        Y = []
        entities = []

        for label, label_vectors in self.vectors_dict.items():
            if label in reverse_dict:
                for v in label_vectors:
                    X.append(v)
                    Y.append(reverse_dict[label])
                    entities.append(label)
        
        save_data_with_pickle(X_PATH, X)
        save_data_with_pickle(Y_PATH, Y)
        save_data_with_pickle(entities_PATH, entities)
                                                                                                                                                                                                                        