from sklearn.model_selection import train_test_split
from collections import Counter
import sys
import time
from collections import defaultdict
from tqdm import tqdm
import random
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from random import sample
import copy
from MTNCI.Filters import Filter, FilterOnClassCohesion

from preprocessing.utils import load_data_with_pickle, save_data_with_pickle
from preprocessing.CorpusManager import CorpusManager
from torch.utils.data import Dataset, DataLoader
import os


class MTNCIDataset(Dataset):

    '''
    This class defines the dataloader for MTNCI
    '''

    def __init__(self, vector_list, label_list, target_list, device):
        '''
        set the values for the dataloader

        params: 
            vector_list: list of vectors which will be the input of the network
            label_list: list of numeric labels which correspond to the label of the input
            target_list: list of dicts, each dict is composed in accord with this project design:
                        {output_denomination: [list of vectors]}
                        in each [list of vectors] the index correspond to the label in label_list
        '''
        self.X = torch.tensor(vector_list, device = device)
        self.labels = torch.tensor(label_list, device = device)
        self.target_list = {k: torch.tensor(x, dtype = torch.float64, device = device) for k, x in target_list.items()}
        
    def __getitem__(self, index):
        '''
        given an index return the input at that index, its label and a dict with the targets

        params: 
            index: the desired index
        returns:
            (an input tensor, its label, a dict {output_denomination: target vector in that space}})
        '''
        return (self.X[index], 
                self.labels[index],
                {k:x[index] for k, x in self.target_list.items()}
            )

    def __len__(self):
        return len(self.X)

class DatasetManager():

    '''
    classes which manage the dataset, inputs are stored in self.X, labels in self.Y and 'entities names' in self.entities 
    '''

    def __init__(self, id):
        '''
        setup the class and the file_id (the id which discriminates the file to load)

        params:
            id: an id which corresponds to the prefix of each file that will be loaded/red
        '''
        self.file_id = id 

    def set_device(self, device):
        '''
        set the default device (cpu/gpu)

        params:
            device: the device (cpu/gpu)
        '''
        self.device = device

    def setup_print_times(self, keywords):
        '''
        setup variables for a future print on loading time
        call this method in couple with print_loaded

        params:
            keywords: the resource to be loaded
        '''

        self.keywords = keywords
        print('-----------------------------')
        print('... loading {} ...'.format(self.keywords))
        self.time = time.time()
    
    def print_loaded(self):
        '''
        print the loading time setupped before 
        call this method in couple with setup_print_times
        '''
        print('{} loaded in {:.2f} seconds'.format(self.keywords, 
                                                   time.time() - self.time))
        self.time = None
        self.keywords = None


    def load_entities_data(self, X_PATH, Y_PATH, ENTITIES_PATH):
        '''
        load the fundamental data X (input), Y (labels), entities (entities names)

        params:
            X_PATH: the path to input vectors
            Y_PATH: the path to labels of input vectors
            ENTITIES_PATH: the path to entities names of input vectors
        '''
        self.setup_print_times(keywords = 'entities data')
        self.X = load_data_with_pickle(X_PATH)
        self.Y = load_data_with_pickle(Y_PATH)
        self.entities = load_data_with_pickle(ENTITIES_PATH)
        self.print_loaded()
    
    def load_concept_embeddings(self, CONCEPT_EMBEDDING_PATHS, nickel = False, cleaned = False):
        '''
        load the concept embedding and transform that in dicts,
        for example: 
            self.concept_embedding is a dict in the format {concept_embedding_name_0: {concept_name_0: vector,
                                                                                       ...
                                                                                       concept_name_N: vector},
                                                                                      },
                                                            concept_embedding_name_1: {concept_name_0: vector,
                                                                                       ...
                                                                                       concetp_name_M: vector},
                                                                                      }
        params:
            CONCEPT_EMBEDDING_PATHS: a list of paths, each path points to a concept embedding                    
        '''

        self.setup_print_times(keywords = 'concept embeddings')
        if not nickel:
            concept_embeddings = [load_data_with_pickle(x) for x in CONCEPT_EMBEDDING_PATHS]

            self.concept_embeddings = {'hyperbolic': concept_embeddings[1]}
        else:
            self.concept_embeddings = {}
            
            self.concept_embeddings['hyperbolic'] = self.load_nickel(CONCEPT_EMBEDDING_PATHS[1])

        self.concept_embeddings['distributional'] = load_data_with_pickle(CONCEPT_EMBEDDING_PATHS[0])
        
        if not cleaned:
        # transform gensim embedding in a dict {key: vector}
            self.concept_embeddings['distributional'] = {k: self.concept_embeddings['distributional'][k] for k in self.concept_embeddings['distributional'].wv.vocab}
        

        self.print_loaded()
    
    def load_nickel(self, path):
        emb = torch.load(path)

        embeddings = emb['embeddings']
        objects = emb['objects']

        return {k: embeddings[objects.index(k)].numpy() for k in objects}

    def split_data_by_unique_entities(self, test_sizes = {'test' : 0.1, 'val': 0.1},
                                            exclude_min_threshold = 10):
        '''
        starting from X, Y and entities creates train, test and validation
            the splitting is based on entity names, because we want to setup a concept invention task,
                so we want to validate and test with UNSEEN ENTITY NAMES in order to see if the network 
                    generalization of the concept is good enough
        
        params:
            test_sizes: a dict which define the relative size of the test and the validation set
            exclude_min_threshold: number of unique entity names under which a concept is excluded
                is setted to 3 because under that number the entity names can't be partitioned in train/test/validation
        
        datasets are stored in X/Y/E_train, X/Y/E_test, X/Y/E_val
        '''

        np.random.seed(236451)

        unique_entities = list(set(self.entities))
        direct_labeling = {e: y for e, y in zip(self.entities, self.Y)}
        labels_of_unique_entities = [direct_labeling[e] for e in unique_entities]
        
        counter = Counter(labels_of_unique_entities)
        
        print('... excluding labels with less than {} unique  entity names...'.format(exclude_min_threshold))
        filtered_entities = [e for e, l in zip(unique_entities, labels_of_unique_entities) if counter[l] >= exclude_min_threshold]
        filtered_labels = [l for l in labels_of_unique_entities if counter[l] >= exclude_min_threshold]
        filtered_out = [l for l in labels_of_unique_entities if counter[l] < exclude_min_threshold]
        
        filtered_out_len = len(set(filtered_out))
        
        print('{} labels filtered out based on exclude_min_threshold ({:.2f}% on dataset dimension)'.format(filtered_out_len,
                                                                                                            len(filtered_out)/len(labels_of_unique_entities)))
        print('Initial labels: {}, current labels: {}'.format(len(set(labels_of_unique_entities)),
                                                              len(set(filtered_labels))))

        print('... splitting in train, test, val ...')
        entities_train, entities_test, \
            y_train_split, y_test_split = train_test_split(filtered_entities,
                                                           filtered_labels,
                                                           test_size = test_sizes['test'] + test_sizes['val'],
                                                           stratify = filtered_labels,
                                                           random_state = 236451)
        
        entities_val, entities_test, \
            y_val, y_test = train_test_split(entities_test,
                                             y_test_split,
                                             test_size = test_sizes['test']/(test_sizes['test'] + test_sizes['val']),
                                             stratify = y_test_split,
                                             random_state = 236451)
                                             
        
        print('... building splitted datasets ...')
        self.X_train = []
        self.Y_train = []
        self.E_train = []

        self.X_val = []
        self.Y_val = []
        self.E_val = []

        self.X_test = []
        self.Y_test = []
        self.E_test = []
        
        bar = tqdm(total = len(self.X))
        filtered_set = set(filtered_entities)
        for x, y, e in zip(self.X, self.Y, self.entities):
            bar.update(1)
            if e in filtered_set:
                if e in entities_train:
                    self.X_train.append(x)
                    self.Y_train.append(y)        
                    self.E_train.append(e)

                elif e in entities_test:
                    self.X_test.append(x)
                    self.Y_test.append(y)
                    self.E_test.append(e)

                elif e in entities_val:
                    self.X_val.append(x)
                    self.Y_val.append(y)
                    self.E_val.append(e)

        
        
    def find_and_filter_not_present(self):
        '''
        finds concept which are not in a concept embedding (for some reasons) and remove those concepts from the dataset
        '''
        print('Initial dataset dimension: ')
        print('X: {}, Y : {}, ENTITIES :{}'.format(len(self.X), len(self.Y), len(self.entities)))
        initial_lengt = len(self.Y)
        not_present = []

        for label in list(set(self.Y)):
            for concept_embedding in self.concept_embeddings.values():
                try:
                    a = concept_embedding[label]
                except:
                    if label not in not_present:
                        not_present.append(label)
        

        self.X = [x for x, y in zip(self.X, self.Y) if y not in not_present]
        self.entities = [e for e, y in zip(self.entities, self.Y) if y not in not_present]

        self.Y = [y for y in self.Y if y not in not_present]

        print('{} concepts are not present in the distributional embedding so are removed'.format(len(not_present)))

        print('dataset dimension after filtering: ')
        print('X: {}, Y : {}, ENTITIES :{}'.format(len(self.X), len(self.Y), len(self.entities)))
        print('{:.2f} % of the dataset is was filtered out'.format(1 - (len(self.Y)/initial_lengt)))

    def shuffle_dataset_and_sample(self, fraction = 1, in_place = True):
        '''
        shuffles the dataset and sample the dataset

        params:
            fraction: the fraction to be sampled
            in_place: if True the samples will substitute the original datast (self.X/Y/entities), otherwise sample are returned
        returns:
            IF in_place == True: sample of X, Y and entities are returned
        '''
        random.seed(236451)
        frac = int(len(self.X) * fraction) 
        print('sampling the {} % of the dataset'.format(fraction * 100))
        A = [[x, y, e] for x, y, e in zip(self.X, self.Y, self.entities)]
        random.shuffle(A)
        if in_place:
            self.X = [x[0] for i, x in enumerate(A) if i < frac]
            self.Y = [x[1] for i, x in enumerate(A) if i < frac]
            self.entities = [x[2] for i, x in enumerate(A) if i < frac]
        else:
            return [x[0] for i, x in enumerate(A) if i < frac], [x[1] for i, x in enumerate(A) if i < frac], [x[2] for i, x in enumerate(A) if i < frac]


    def save_raw_dataset(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        save_data_with_pickle(save_path + 'X', self.X)
        save_data_with_pickle(save_path + 'Y', self.Y)
        save_data_with_pickle(save_path + 'entities', self.entities)

    def load_raw_dataset(self, load_path):
        self.X = load_data_with_pickle(load_path + 'X')
        self.Y = load_data_with_pickle(load_path + 'Y')
        self.entities = load_data_with_pickle(load_path + 'entities')

    def save_datasets(self, save_path):
        '''
        saves datasets in the save_path

        params:
            save_path: the path in which save datasets
        '''

        print(save_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        save_data_with_pickle(save_path + 'filtered_X_train', self.X_train)
        save_data_with_pickle(save_path + 'filtered_X_val', self.X_val)
        save_data_with_pickle(save_path + 'filtered_X_test', self.X_test)

        save_data_with_pickle(save_path + 'filtered_Y_train', self.Y_train)
        save_data_with_pickle(save_path + 'filtered_Y_val', self.Y_val)
        save_data_with_pickle(save_path + 'filtered_Y_test', self.Y_test)

        save_data_with_pickle(save_path + 'filtered_entities_train', self.E_train)
        save_data_with_pickle(save_path + 'filtered_entities_val', self.E_val)
        save_data_with_pickle(save_path + 'filtered_entities_test', self.E_test)


    def print_statistic_on_dataset(self):
        '''
        print some statistics on the classes distribution in the datasets
        '''

        Tr = Counter(self.Y_train).most_common()
        Va = Counter(self.Y_val).most_common()
        Te = Counter(self.Y_test).most_common()

        print('{:^30}|{:^30}|{:^30}'.format('Train','Val', 'Test'))
        print('{:-^30}|{:-^30}|{:-^30}'.format('', '', ''))

        for x, y, z in zip(Tr, Va, Te):
            print('{:^25}{:^5}|{:^25}{:^5}|{:^25}{:^5}'.format(x[0], x[1], y[0], y[1], z[0], z[1]))


    def plot_datasets(self):
        '''
        generate a barplot on the classes distribution in the datasets
        '''
        Tr = Counter(self.Y_train).most_common()
        Va = Counter(self.Y_val).most_common()
        Te = Counter(self.Y_test).most_common()

        Tr = {t[0]: t[1] for t in Tr}
        Va = {t[0]: t[1] for t in Va}
        Te = {t[0]: t[1] for t in Te}

        # set width of bar
        barWidth = 0.25
        
        # set height of bar
        bars1 = [t for t in Tr.values()]
        bars2 = [Va[k] for k in Tr.keys()]
        bars3 = [Te[k] for k in Tr.keys()]
        
        # Set position of bar on X axis
        r1 = np.arange(len(Tr))
        r2 = [x + barWidth for x in r1]
        r3 = [x + barWidth for x in r2]
        
        # Make the plot
        plt.figure(figsize=(25, 25))
        plt.bar(r1, bars1, color='#7f6d5f', width=barWidth, edgecolor='white', label='Train')
        plt.bar(r2, bars2, color='#557f2d', width=barWidth, edgecolor='white', label='Validation')
        plt.bar(r3, bars3, color='#2d7f5e', width=barWidth, edgecolor='white', label='Test')
        
        # Add xticks on the middle of the group bars
        plt.xlabel(' # of vectors x classes different dataset ', fontweight='bold')
        plt.xticks([r + barWidth for r in range(len(bars1))], list(Tr.keys()), rotation = 75)
        plt.yscale('log')
        
        # Create legend & Show graphic
        plt.legend()
        plt.savefig('./PLOT.png')

    def get_numeric_label(self, labels):
        '''
        converts explicit labels (concepts names) in numbers
        this method is based on self.numeric_label_map which is computed in compute_numeric_labels

        params:
            labels: a list of labels which will be converted 
        return:
            a list of numbers which corresponds to input labels
        '''
        return [self.numeric_label_map[y] for y in labels]

    def compute_numeric_labels(self):
        '''
        computes a map and an inverse map which will be used to map labels to numbers (requested by pytorch)
        '''
        self.numeric_label_map = {y: x for x, y in enumerate(set(self.Y_train))}
        self.inverse_numeric_label_map = {v:k for k,v in self.numeric_label_map.items()}

    def create_numeric_dataset(self):
        '''
        map the Y datasets to numeric datasets using get_numeric_label function on each label in Y
        '''
        self.compute_numeric_labels()
        self.Y_numeric_train = self.get_numeric_label(self.Y_train)
        self.Y_numeric_test = self.get_numeric_label(self.Y_test)
        self.Y_numeric_val = self.get_numeric_label(self.Y_val)

    def create_aligned_dataset(self, in_place = True):
        '''
        create datasets which align input vectors to a dict of output vectors

        params:
            in_place: if = True saves the aligned datasets in self.aligned_y_*
                         = False returns the aligned datasets
        
        returns:
            if in_place == False returns the aligned train, test and validation dataset
        '''

        aligned_y_train = {k:[] for k in self.concept_embeddings.keys()}
        aligned_y_test = {k:[] for k in self.concept_embeddings.keys()}
        aligned_y_val = {k:[] for k in self.concept_embeddings.keys()}

        for label in self.Y_train:
            for k, emb in self.concept_embeddings.items():
                aligned_y_train[k].append(emb[label])
        
        for label in self.Y_test:
            for k, emb in self.concept_embeddings.items():
                aligned_y_test[k].append(emb[label])

        for label in self.Y_val:
            for k, emb in self.concept_embeddings.items():
                aligned_y_val[k].append(emb[label])
        
        
        for k, dataSET in aligned_y_train.items():
            aligned_y_train[k] = np.array(dataSET)
        
        for k, dataSET in aligned_y_test.items():
            aligned_y_test[k] = np.array(dataSET)
        
        for k, dataSET in aligned_y_val.items():
            aligned_y_val[k] = np.array(dataSET)
        
        if in_place:
            self.aligned_y_train = aligned_y_train
            self.aligned_y_test = aligned_y_test
            self.aligned_y_val = aligned_y_val
        else:
            return aligned_y_train, aligned_y_test, aligned_y_val

    
    def create_dataloaders(self, batch_sizes = {'train': 512, 'test': 512, 'val': 512}):
        '''
        generates dataloaders and save on self.
        '''

        trainset = MTNCIDataset(self.X_train,
                                self.Y_numeric_train,
                                self.aligned_y_train,
                                device = self.device) 

        self.trainloader = DataLoader(trainset, batch_size=batch_sizes['train'], shuffle=True)

        testset = MTNCIDataset(self.X_test,
                               self.Y_numeric_test,
                               self.aligned_y_test,
                               device = self.device) 
        self.testloader = DataLoader(testset, batch_size=batch_sizes['test'], shuffle=False)

        valset = MTNCIDataset(self.X_val,
                              self.Y_numeric_val,
                              self.aligned_y_val,
                              device = self.device)
        self.valloader = DataLoader(valset, batch_size=batch_sizes['val'], shuffle=True)   

    def compute_weights(self):
        '''
        compute weights on labels
        '''
        self.weights = {}

        for vectors, label in zip([self.Y_numeric_train, self.Y_numeric_val, self.Y_numeric_test],
                                 ['Train', 'Val', 'Test']):
            
            # x_unique = torch.tensor(vectors, device=self.device).unique(sorted=True)
            # x_unique_count = torch.stack([(torch.tensor(vectors, device=self.device)==x_u).sum() for x_u in x_unique]).float()
            # m = torch.max(x_unique_count)
            # labels_weights = m/x_unique_count

            x_unique_count = Counter(set(vectors.detach().cpu().numpy()))
            m = max(x_unique_count.values())
            labels_weights = [m/x_unique_count[i] if x_unique_count[i] != 0 else 0 for i in range(max(x_unique_count.keys()) + 1)]
            labels_weights = torch.tensor(labels_weights, device=self.device)
            self.weights[label] = labels_weights / torch.norm(labels_weights)
    
    def get_weights(self, label):
        '''
        returns the weight of the input label, this method needs to be called after the compute_weights method 

        params:
            label: the label on which get the weigth
        
        return:
            the weigth of the input label
        '''
        return self.weights[label]


    def normalize(self):
        '''
        normalize the input vectors 
        '''
        print('... normalization ...')
        self.X = normalize(self.X, axis = 1)



    def setup_filter(self, filter_name, log_file_path, filtered_dataset_path, cluster_distance = cosine_similarity,
                            X = None, Y = None, entities = None, selfXY = True, 
                            threshold = 0.6, quantile = 0.05):
        '''
        setup the filter which will be used to filter input vectors (you don't say)

        params:
            filter_name : the name of the filter, has to be a value in Filter().FILTERS
            log_file_path: path to the log file
            filtered_dataset_path: path in which the filtered will be saved
            X, Y and entities: the dataset to be filtered (requires selfXY = False) 
            selfXY: if = True: setup self.X/Y/entities as dataset to be filtered
        '''
        self.selected_filter = Filter().select_filter(filter_name)
        
        self.selected_filter.set_parameters(log_file_path = log_file_path, 
                                            filtered_dataset_path = filtered_dataset_path,
                                            threshold=threshold, quantile = quantile, cluster_distance = cluster_distance)

        if selfXY:
            self.selected_filter.set_data(X = self.X, Y = self.Y, entities = self.entities)
        elif X and Y and entities:
            self.selected_filter.set_data(X = X, Y = Y, entities = entities)
        else:
            raise Exception('Error: pass an X and an Y or set selfXY to true!!!') from e

    def get_epoch_data(self, dataset=None, initialize = False, batch_iteration = 0):
        if initialize:

            self.train_it = iter(self.trainloader)
            self.val_it = iter(self.valloader)
            
        elif dataset == 'train':
            return next(self.train_it)

        elif dataset == 'val':
            return next(self.val_it)
        else:
            raise Exception('please pass a valid dataset name') from e
    
    def get_data_batch_length(self, dataset):
        if dataset == 'train':
            return len(self.trainloader)
        elif dataset == 'val':
            return len(self.valloader)
        else:
            raise Exception('please pass a valid dataset name') from e

    
    def filter(self):
        '''
        launch the filter method on the selected filter
        '''
        self.X, self.Y, self.entities = self.selected_filter.filter()
           
class ShimaokaMTNCIDatasetManager(DatasetManager):

    def get_data_batch_length(self, dataset):
        if dataset == 'train':
            return len(self.train_batched_datas)
        elif dataset == 'val':
            return len(self.val_batched_datas)
        else:
            raise Exception('please pass a valid dataset name') from e

    def get_epoch_data(self, dataset=None, batch_iteration = 0, initialize = False):
        if initialize:
            pass
        else:
            if dataset == 'train':
                x = self.train_batched_datas[batch_iteration]
                labels = self.batched_Y_train[batch_iteration]
                targets = self.batched_aligned_train[batch_iteration][0]
                
            elif dataset == 'val':
                x = self.val_batched_datas[batch_iteration]
                labels = self.batched_Y_val[batch_iteration]
                targets = self.batched_aligned_val[batch_iteration][0]
                
            else:
                raise Exception('please pass a valid dataset name') from e
            
            return x, labels, targets
            

    def set_batched_data(self, train, val, test):
        self.Y_train = train['labels']
        self.Y_val = val['labels']
        self.Y_test = test['labels']
        

        self.train_batched_datas = train['data']
        self.val_batched_datas = val['data']
        self.test_batched_datas = test['data']


        self.train_batched_labels = self.batch_labels(train)
        self.val_batched_labels = self.batch_labels(val)
        self.train_batched_labels = self.batch_labels(test)

        self.create_numeric_dataset()

        self.Y_numeric_train = torch.tensor(self.Y_numeric_train, device = self.device)
        self.Y_numeric_val = torch.tensor(self.Y_numeric_val, device = self.device)
        self.Y_numeric_test = torch.tensor(self.Y_numeric_test, device = self.device)
        
        n_train = {'data': train['data'], 'labels' : self.Y_numeric_train}
        n_val = {'data': val['data'], 'labels' : self.Y_numeric_val}
        n_test = {'data': test['data'], 'labels' : self.Y_numeric_test}

        self.batched_Y_train = self.batch_labels(n_train)
        self.batched_Y_val = self.batch_labels(n_val)
        self.batched_Y_test = self.batch_labels(n_test)

        
        self.create_aligned_dataset()

        self.aligned_y_train = {k: torch.tensor(v, device=self.device) for k, v in self.aligned_y_train.items()}
        self.aligned_y_val = {k: torch.tensor(v, device=self.device) for k, v in self.aligned_y_val.items()}
        self.aligned_y_test = {k: torch.tensor(v, device=self.device) for k, v in self.aligned_y_test.items()}

        n_train = {'data': train['data'], 'labels' : self.aligned_y_train}
        n_val = {'data': val['data'], 'labels' : self.aligned_y_val}
        n_test = {'data': test['data'], 'labels' : self.aligned_y_test}
        
        self.batched_aligned_train = self.batch_labels(n_train, lisst=False)
        self.batched_aligned_val = self.batch_labels(n_val, lisst=False)
        self.batched_aligned_test = self.batch_labels(n_test, lisst=False)
    
    def batch_labels(self, data, lisst = True):
        batched = []
        if lisst:
            length = len(data['labels'])
        else:
            length = len(data['labels']['distributional'])

        offset = len(data['data'][0][0])
        for i in range(0, length, offset):
            if lisst:
                batched.append(data['labels'][i: i + offset])
            else:
                batched.append([{'distributional': data['labels']['distributional'][i: i + offset],
                                'hyperbolic': data['labels']['hyperbolic'][i: i + offset]}])
        
        return batched

class ChoiDatasetManager(ShimaokaMTNCIDatasetManager):

    def load_concept_embeddings(self, CONCEPT_EMBEDDING_PATHS, data):
        '''
        load the concept embedding and transform that in dicts,
        for example: 
            self.concept_embedding is a dict in the format {concept_embedding_name_0: {concept_name_0: vector,
                                                                                       ...
                                                                                       concept_name_N: vector},
                                                                                      },
                                                            concept_embedding_name_1: {concept_name_0: vector,
                                                                                       ...
                                                                                       concetp_name_M: vector},
                                                                                      }
        params:
            CONCEPT_EMBEDDING_PATHS: a list of paths, each path points to a concept embedding                    
        '''

        self.setup_print_times(keywords = 'concept embeddings')

        self.concept_embeddings = {}
        self.concept_embeddings['hyperbolic'] = self.load_nickel(CONCEPT_EMBEDDING_PATHS[1])
        self.concept_embeddings['distributional'] = self.load_nickel(CONCEPT_EMBEDDING_PATHS[1])
        
        self.print_loaded()

    def set_batched_data(self, train, val, test):
        self.Y_train = train['labels']
        self.Y_val = val['labels']
        self.Y_test = test['labels']


        self.all_labels = set()

        for d in [self.Y_train, self.Y_val, self.Y_test]:
            for y in d:
                for label in y:
                    self.all_labels.add(label)
        
        self.train_batched_datas = train['data']
        self.val_batched_datas = val['data']
        self.test_batched_datas = test['data']

        self.train_batched_labels = self.batch_labels(train)
        self.val_batched_labels = self.batch_labels(val)
        self.train_batched_labels = self.batch_labels(test)

        self.create_numeric_dataset()


        self.Y_numeric_train = torch.tensor(self.Y_numeric_train, device = self.device)
        self.Y_numeric_val = torch.tensor(self.Y_numeric_val, device = self.device)
        self.Y_numeric_test = torch.tensor(self.Y_numeric_test, device = self.device)
        
        n_train = {'data': train['data'], 'labels' : self.Y_numeric_train}
        n_val = {'data': val['data'], 'labels' : self.Y_numeric_val}
        n_test = {'data': test['data'], 'labels' : self.Y_numeric_test}

        self.batched_Y_train = self.batch_labels(n_train)
        self.batched_Y_val = self.batch_labels(n_val)
        self.batched_Y_test = self.batch_labels(n_test)

        self.create_aligned_dataset()

        self.aligned_y_train = {k: torch.tensor(v, device=self.device) for k, v in self.aligned_y_train.items()}
        self.aligned_y_val = {k: torch.tensor(v, device=self.device) for k, v in self.aligned_y_val.items()}
        self.aligned_y_test = {k: torch.tensor(v, device=self.device) for k, v in self.aligned_y_test.items()}

        n_train = {'data': train['data'], 'labels' : self.aligned_y_train}
        n_val = {'data': val['data'], 'labels' : self.aligned_y_val}
        n_test = {'data': test['data'], 'labels' : self.aligned_y_test}
        
        self.batched_aligned_train = self.batch_labels(n_train, lisst=False)
        self.batched_aligned_val = self.batch_labels(n_val, lisst=False)
        self.batched_aligned_test = self.batch_labels(n_test, lisst=False)

    def create_numeric_dataset(self):
        '''
        map the Y datasets to numeric datasets using get_numeric_label function on each label in Y
        '''
        self.compute_numeric_labels()

        self.max_label_input = max([len(y) for y in self.Y_train])

        self.Y_numeric_train = self.get_numeric_label(self.Y_train)
        self.Y_numeric_test = self.get_numeric_label(self.Y_test)
        self.Y_numeric_val = self.get_numeric_label(self.Y_val)

    def compute_numeric_labels(self):
        '''
        computes a map and an inverse map which will be used to map labels to numbers (requested by pytorch)
        '''
        self.numeric_label_map = {y: x for x, y in enumerate(set(self.all_labels))}
        self.inverse_numeric_label_map = {v:k for k,v in self.numeric_label_map.items()}
    
    def get_numeric_label(self, labels_dataset):
        
        numeric_dataset = []
        for labels in labels_dataset:
            numeric_labels = []
            for y in labels:
                numeric_labels.append(self.numeric_label_map[y])
            
            while len(numeric_labels) < self.max_label_input:
                numeric_labels.append(-1)
            numeric_dataset.append(numeric_labels)

        return numeric_dataset
    
    def create_aligned_dataset(self, in_place = True):
        '''
        create datasets which align input vectors to a dict of output vectors

        params:
            in_place: if = True saves the aligned datasets in self.aligned_y_*
                         = False returns the aligned datasets
        
        returns:
            if in_place == False returns the aligned train, test and validation dataset
        '''

        aligned_y_train = {k:[] for k in self.concept_embeddings.keys()}
        aligned_y_test = {k:[] for k in self.concept_embeddings.keys()}
        aligned_y_val = {k:[] for k in self.concept_embeddings.keys()}

        for label in self.Y_train:
            for k, emb in self.concept_embeddings.items():
                aligned_y_train[k].append(self.get_vectors_from_embedding(embedding = emb, labels = label))
        
        for label in self.Y_test:
            for k, emb in self.concept_embeddings.items():
                aligned_y_test[k].append(self.get_vectors_from_embedding(embedding = emb, labels = label))

        for label in self.Y_val:
            for k, emb in self.concept_embeddings.items():
                aligned_y_val[k].append(self.get_vectors_from_embedding(embedding = emb, labels = label))
        

        for k, dataSET in aligned_y_train.items():
            aligned_y_train[k] = np.array(dataSET)
        
        for k, dataSET in aligned_y_test.items():
            aligned_y_test[k] = np.array(dataSET)
        
        for k, dataSET in aligned_y_val.items():
            aligned_y_val[k] = np.array(dataSET)
        

        if in_place:
            self.aligned_y_train = aligned_y_train
            self.aligned_y_test = aligned_y_test
            self.aligned_y_val = aligned_y_val
        else:
            return aligned_y_train, aligned_y_test, aligned_y_val

    def get_vectors_from_embedding(self, embedding, labels):
        vectors = []
        for l in labels:
            vectors.append(embedding[l])

        i = 0
        while len(vectors) < self.max_label_input:
            vectors.append(vectors[i % len(labels)])
            i += 1
        return vectors