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
from abc import ABC, abstractmethod
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from random import sample
import copy


from preprocessing.utils import load_data_with_pickle, save_data_with_pickle
from preprocessing.CorpusManager import CorpusManager
from torch.utils.data import Dataset, DataLoader

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

    def load_concept_embeddings(self, CONCEPT_EMBEDDING_PATHS):
        '''
        load the concept embedding and transform that in dicts,
        for example: 
            self.concept_embedding is a dict in the format {concept_embedding_name_0: {concept_name_0: vector,
                                                                                       ...
                                                                                       concetp_name_N: vector},
                                                                                      },
                                                            concept_embedding_name_1: {concept_name_0: vector,
                                                                                       ...
                                                                                       concetp_name_M: vector},
                                                                                      }
        params:
            CONCEPT_EMBEDDING_PATHS: a list of paths, each path points to a concept embedding                    
        '''

        self.setup_print_times(keywords = 'concept embeddings')

        concept_embeddings = [load_data_with_pickle(x) for x in CONCEPT_EMBEDDING_PATHS]

        self.concept_embeddings = {'hyperbolic': concept_embeddings[1],
                                   'distributional':concept_embeddings[0]}
        # transform gensim embedding in a dict {key: vector}
        self.concept_embeddings['distributional'] = {k: self.concept_embeddings['distributional'][k] for k in self.concept_embeddings['distributional'].wv.vocab}
        # concept_embeddings['hyperbolic'] = concept_embeddings['hyperbolic'][0]
        self.print_loaded()
        
    def split_data_by_unique_entities(self, test_sizes = {'test' : 0.1, 'val': 0.1},
                                            exclude_min_threshold = 3):
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
                                                           stratify = filtered_labels)
        
        entities_val, entities_test, \
            y_val, y_test = train_test_split(entities_test,
                                             y_test_split,
                                             test_size = test_sizes['test']/(test_sizes['test'] + test_sizes['val']),
                                             stratify = y_test_split)
        
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

    def save_datasets(self, save_path, ID):
        '''
        saves datasets in the save_path using ID as prefix

        params:
            save_path: the path in which save datasets
            ID: the prefix that will be used on saved datasets names
        
        '''
        save_data_with_pickle(save_path + ID + 'filtered_X_train', self.X_train)
        save_data_with_pickle(save_path + ID + 'filtered_X_val', self.X_val)
        save_data_with_pickle(save_path + ID + 'filtered_X_test', self.X_test)

        save_data_with_pickle(save_path + ID + 'filtered_Y_train', self.Y_train)
        save_data_with_pickle(save_path + ID + 'filtered_Y_val', self.Y_val)
        save_data_with_pickle(save_path + ID + 'filtered_Y_test', self.Y_test)

        save_data_with_pickle(save_path + ID + 'filtered_entities_train', self.E_train)
        save_data_with_pickle(save_path + ID + 'filtered_entities_val', self.E_val)
        save_data_with_pickle(save_path + ID + 'filtered_entities_test', self.E_test)


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
            x_unique = torch.tensor(vectors, device=self.device).unique(sorted=True)
            x_unique_count = torch.stack([(torch.tensor(vectors, device=self.device)==x_u).sum() for x_u in x_unique]).float()
            m = torch.max(x_unique_count)
            labels_weights = m/x_unique_count
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
        self.X = normalize(self.X, axis = 1)


    def setup_filter(self, filter_name, log_file_path, filtered_dataset_path, X = None, Y = None, entities = None, selfXY = True):
        '''
        setup the filter which will be used to filter input vectors (you don't say)

        params:
            filter_name : the name of the filter, has to be a value in Filter().FILTERS
            log_file_path: path to the log file
            filtered_dataset_path: path in which the filtered will be saved
            X, Y and entities: the dataset to be filtered (requires selfXY = False) 
            selfXY: if = True: setup self.X/Y/entities as dataset to be filtered
        '''
        self.selected_filter = Filter().select_filter(filter_name, log_file_path, filtered_dataset_path)
        
        self.selected_filter.set_parameters()

        if selfXY:
            self.selected_filter.set_data(X = self.X, Y = self.Y, entities = self.entities)
        elif X and Y and entities:
            self.selected_filter.set_data(X = X, Y = Y, entities = entities)
        else:
            raise Exception('Error: pass an X and an Y or set selfXY to true!!!') from e
    
    def filter(self):
        '''
        launch the filter method on the selected filter
        '''
        self.X, self.Y, self.entities = self.selected_filter.filter()


class Filter():

    '''
    parent class of each filter, defines the base methods
    '''

    def __init__(self, log_file_path = None, filtered_dataset_path = None):
        '''
        inizialize some values, 
        declare in FILTERS the name of all filter
        '''
        self.FILTERS = {'ClassCohesion': 'CC'}
        self.log_file = log_file_path
        self.filtered_dataset_path = filtered_dataset_path
    
    def set_data(self, X, Y, entities):
        '''
        set the data to filter

        params:
            X: the input to filter
            Y: the labels of the inputs
            entities: the entity names of the input
        '''
        self.X = X
        self.Y = Y
        self.entities = entities

    def log(self, text):
        '''
        write the text on the log
        
        params:
            text: the text to be logged
        '''
        with open(self.log_file, 'a') as out:
            out.write(text + '\n')

    def select_filter(self, filter_name, log_file_path, filtered_dataset_path):
        '''
        select the filter based on filter_name

        params:
            filter_name: the name of the filter
            log_file_path: the path to the log file
            filtered_dataset_path: the path in which save the the filtered dataset
        '''
        if filter_name == self.FILTERS['ClassCohesion']:
            return FilterOnClassCohesion(log_file_path = log_file_path, 
                                         filtered_dataset_path = filtered_dataset_path)

    @abstractmethod
    def filter():
        pass

    @abstractmethod
    def save_data():
        pass


class FilterOnClassCohesion(Filter):

    def save_data(self):
        '''
        save the filtered dataset to the filreted_dataset_path
        '''
        if not (len(self.X) == len(self.Y) and len(self.Y) == len(self.entities)):
            raise Exception('Error in filtering process, datasets are not aligned')

        save_data_with_pickle(self.filtered_dataset_path + 'X', self.X)
        save_data_with_pickle(self.filtered_dataset_path + 'Y', self.Y)
        save_data_with_pickle(self.filtered_dataset_path + 'entities', self.entities)

    def log_out_words(self, out_words):
        '''
        logs the words filtered out by the filtering process

        params:
            out_words: a dict {concept: [list of words]} which will be logged 
        '''
        with open('../source_files/logs/out_words_log.txt', 'a') as out:
            out.write('\n ------------------------------------------------------ \n')
            for concept, words in sorted(out_words.items()):
                out.write('{}: {} words filtered, {} unique words\n'.format(concept, len(words), len(set(words))))
                c = sorted(Counter(words).items())
                for couple in c:
                    out.write('\t\t {}: {}\n'.format(couple[0], couple[1]))


    def log_in_words(self, in_words):
        '''
        logs the maintained words by the filtering process

        params:
            in_words: a dict {concept: [list of words]} which will be logged 
        '''
        with open('../source_files/logs/in_words_log.txt', 'a') as out:
            out.write('\n ------------------------------------------------------ \n')
            for concept, words in sorted(in_words.items()):
                out.write('{}: {} words maintained, {} unique words\n'.format(concept, len(words), len(set(words))))
                c = sorted(Counter(words).items())
                for couple in c:
                    out.write('\t\t {}: {}\n'.format(couple[0], couple[1]))

    def filter_dataset_on_numerosity(self, filtering_treshold):
        '''
        filters the dataset based on the numerosity of concept vectors, 
            if a concept has less than filtering_treshold vectors will be filtered out
        '''
        self.entities_dataset = {k: vs for k, vs in self.entities_dataset.items() if len(self.dataset[k]) > filtering_treshold} 
        self.dataset = {k: vs for k, vs in self.dataset.items() if len(vs) > filtering_treshold}
        
    def filter(self):
        '''
        define the pipeline of the filter
        '''

        filtering_treshold = 100
        
        print('--- FILTERING PROCESS ---')
        print('... creating dataset ...')

        self.create_datasets()
        self.filter_dataset_on_numerosity(filtering_treshold)
        print('... creating clusters ...')
        self.clusters()

        print('... computing sampled silhouette ...')
        silh = self.sampled_silhouette()

        self.log('silhouette score: {}'.format(silh))

        print('... filtering out quantiles until {} cohesion is reached ... '.format(self.threshold))
        filtered_dataset, filtered_entities_dataset, out_words = self.filter_quantile(threshold = self.threshold, 
                                                                                      quantile=self.quantile,
                                                                                    filtering_treshold=filtering_treshold)
        filtered_dataset = {k: vs for k, vs in filtered_dataset.items() if len(vs) > filtering_treshold}
        filtered_entities_dataset = {k: entities for k, entities in filtered_entities_dataset.items() if len(entities) > filtering_treshold}

        self.log_out_words(out_words)
        self.log_in_words(filtered_entities_dataset)

        print('vectors in filtered dataset:{}'.format(len([v for k, vs in filtered_dataset.items() for v in vs])))
        self.log('vectors in filtered dataset:{}'.format(len([v for k, vs in filtered_dataset.items() for v in vs])))
        
        self.dataset = filtered_dataset
        self.entities_dataset = filtered_dataset

        print('... re-creating clusters ...')
        self.clusters()

        print('... re-computing sampled silhouette ...')
        silh = self.sampled_silhouette()
        self.log('silhouette score on filtered dataset: {}'.format(silh))

        # self.save_data()

        X = [vector for concept, vectors in self.dataset.items() for vector in vectors]
        Y = [concept for concept, vectors in self.dataset.items() for vector in vectors]
        entities = [entity for concept, entities in self.entities_dataset.items() for entity in entities]

        return X, Y, entities


    def clusters(self):
        '''
        create the clusters sizes of the dataset, clean the returns and log the cohesions, 
        '''
        self.cluster_sizes, cohesions = self.create_clusters()

        self.cohesions = {k: np.mean(v) for k, v in cohesions.items()}

        self.log(self.get_stats(cohesions = self.cohesions))

    def sampled_silhouette(self):
        '''
        compute the silhouted based on representative samples of each concept

        returns:
            silhouette_score: the silhouette value
        '''
        X = [vector for concept, vectors in self.dataset.items() for vector in vectors]
        Y = [concept for concept, vectors in self.dataset.items() for vector in vectors]

        print('number of vectors in the dataset: {}'.format(len(Y)))
        self.log('number of vectors in the dataset: {}'.format(len(Y)))
        
        dataset = {y:[] for y in set(Y)}
        for x, y in zip(X, Y):
            dataset[y].append(x)
        
        sampled_dataset = {y: sample(dataset[y], self.cluster_sizes[y]) for y in set(Y)}
        sampled_Y = [label for label, vectors in sampled_dataset.items() for x in sampled_dataset[label]]
        sampled_X = [x for label, vectors in sampled_dataset.items() for x in sampled_dataset[label]]
        
        print('number of vectors in the sampled dataset: {}'.format(len(sampled_Y)))
        self.log('number of vectors in the sampled dataset: {}'.format(len(sampled_Y)))
        
        return silhouette_score(sampled_X, sampled_Y, metric='cosine')

    def set_parameters(self, threshold = 0.5, quantile = 0.05):
        '''
        set the filters parameters

        params:
            treshold: the coherence to be reached by each concept cluster
            quantile: the quantile which will be removed on each iteration until the treshold is not reached 
        '''
        self.threshold = threshold
        self.quantile = quantile

    def create_datasets(self):
        '''
        create the dataset which will be used in all the pipeline
        '''
        self.dataset = defaultdict(list)
        self.entities_dataset = defaultdict(list)

        for x, y, e in zip(self.X, self.Y, self.entities):
            self.dataset[y].append(x)
            self.entities_dataset[y].append(e)

        
        self.dataset = {k:values for k, values in self.dataset.items() if len(values) > 1}
        self.entities_dataset = {k:entities for k, entities in self.entities_dataset.items() if len(entities) > 1}


    def create_clusters(self, epsilon = 0.005, iterations = 5, patience_threshold = 10, max_size = 3000):
        '''
        for each concept compute the optimal size for a representative sample

        params:
            epsilon: the value of changement of mean coherence and std between one iteration sample and the successive 
            iterations: the number of sample used to compute the mean coherence and the standard deviation
            patience_treshold: the number of consecutive times in which both mean and standard deviant are lesser than the epsilon value
            max_size: the maximum size of a cluster

        returns:
            cluster_sizes: the optimal size of each cluster
            cohesions: the cohesion value of each cluster
        '''
        
        print('vectors in dataset:{}'.format(len([v for k, vs in self.dataset.items() for v in vs])))
        
        distances = defaultdict(list)
        xs = defaultdict(list)
        cluster_sizes = {}
        cohesions = {}

        tot = len(self.dataset)
        
        T = time.time()

        for i, (k, v) in enumerate(self.dataset.items()):

            sample_size = 200

            last_dist = 0    
            difference = 1

            t = time.time()

            patience = 0
            last_meann = 0

            while patience < patience_threshold and sample_size < max_size:

                cohesions[k] = []

                for j in range(iterations):

                    if sample_size < len(v):
                        sam = sample(v, sample_size)
                    else:
                        sam = v

                    xs[k].append(len(sam))
                    
                    cohesion = np.sum(np.tril(cosine_similarity(sam), -1))/sum([I + 1 for I in range(len(sam) - 1)])
                    cohesions[k].append(cohesion)
                    

                meann = np.mean(cohesions[k])

                difference_on_mean = abs(meann - last_meann)

                last_meann = meann

                standev = np.std(cohesions[k])

                if standev < epsilon and difference_on_mean < epsilon:
                    patience += 1
                else:
                    patience = 0

                sample_size += 20

            cluster_sizes[k] = len(sam)

            output = '{:3}/{}: {:>6}/{:<6}, {:.2f} , {:6} seconds, {}'.format(i + 1, tot, len(sam), len(self.dataset[k]), meann, round(time.time() - t, 3), k)
            print(output)
            self.log(text = output)
            t = time.time()
            
        print('total_time: {}'.format(round(time.time() - T, 3)))
        self.log(text = 'total_time: {}'.format(round(time.time() - T, 3)))
        
        return cluster_sizes, cohesions

    def filter_quantile(self, threshold, filtering_treshold = 2, quantile = 0.05):
        
        '''
        for each concept, create a sample of the cluster size, compute the cohesion of each vector w.r.t the sample
        and iterative exclude all quantiles, re-sample and recompute the cohesion until the threshold is reached 

        params:
            threshold: the cohesion value to be reached from each cluster
            filtering_treshold: the minimum number of value in each cluster
            quantile: the quantile to remove at each iteration

        returns:
            dataset_: the filtered dataset
            entities_dataset_: the entities names of the filtered dataset
            out_words: the filtered out entities names
        '''

        dataset_ = copy.deepcopy(self.dataset)
        entities_dataset_ = copy.deepcopy(self.entities_dataset)
        cohesions_ = copy.deepcopy(self.cohesions)
        
        filtered_dataset = {}
        filtered_entities_dataset = {}

        out_words = {k:[] for k, _ in dataset_.items()}

        filtered_quantities = {k:0 for k in dataset_}
        tot = len(dataset_)
        
        initial_length = {k: len(v) for k, v in dataset_.items()}
        
        t = time.time()
        first = True
        filtered_cohesions = [0]
        
        while min(filtered_cohesions) < threshold:
            
            filtered_cohesions = []
            filtered_dataset = {k:[] for k, _ in dataset_.items()}
            filtered_entities_dataset = {k: [] for k, _ in dataset_.items()}
            while_time = time.time()
            
            for i, ((k, vs), (_, entities)) in enumerate(zip(dataset_.items(), entities_dataset_.items())):
                if len(vs) != len(entities):
                    raise Exception('ERROR IN VECTOR-ENTITIES ALIGNMENT!!') from e
                if cohesions_[k] < threshold:
                    if self.cluster_sizes[k] < len(vs):
                        class_cluster = sample(vs, self.cluster_sizes[k])
                    else:
                        class_cluster = vs
                    
                    if len(class_cluster) >= filtering_treshold:

                        computed_cohesions = np.mean(cosine_similarity(vs, class_cluster), axis=1)
                        q = np.quantile(computed_cohesions, quantile, interpolation='nearest')

                        computed_cohesion_mask = np.where(computed_cohesions <= q, 0, 1)

                        for j, (v, e) in enumerate(zip(vs, entities)):
                            if computed_cohesion_mask[j]:
                                filtered_dataset[k].append(v)
                                filtered_entities_dataset[k].append(e)
                            else:
                                filtered_quantities[k] += 1
                                out_words[k].append(e)

                        if len(filtered_dataset[k]) > filtering_treshold:
                            if self.cluster_sizes[k] < len(filtered_dataset[k]):
                                class_cluster = sample(filtered_dataset[k], self.cluster_sizes[k])
                            else:
                                class_cluster = filtered_dataset[k]
                    
                            if len(filtered_dataset[k]) != len(class_cluster):
                                new_cohesion = np.sum(cosine_similarity(filtered_dataset[k], class_cluster))/(len(filtered_dataset[k] * len(class_cluster)))
                            else:
                                new_cohesion = np.sum(np.tril(cosine_similarity(filtered_dataset[k]), -1))/sum([I + 1 for I in range(len(filtered_dataset[k]) - 1)])
                                                                            
                            filtered_cohesions.append(new_cohesion)
                            
                            cohesions_[k] = new_cohesion
                            verbose = True
                            output = '    {:3}/{}: {:>6}/{:<6}, {:.3f} seconds, cohesion:{:.4f}, {}   '.format(i + 1, 
                                                                                                tot, 
                                                                                                filtered_quantities[k],
                                                                                                initial_length[k],
                                                                                                time.time() - t,
                                                                                                new_cohesion,
                                                                                                k)
                            

                            
                        else:
                            filtered_dataset[k] = []
                            filtered_entities_dataset[k] = []
                            verbose = False
                            output = ' xx {:3}/{}: {} has not passed the filter process (no more vectors) xx'.format(i + 1, 
                                                                                                              tot,
                                                                                                              k)

                    else:
                        filtered_dataset[k] = []
                        filtered_entities_dataset[k] = []
                        verbose = False
                        output = ' xx {:3}/{}: {} has not passed the filter process (no more vectors) xx'.format(i + 1, 
                                                                                                          tot,
                                                                                                          k)
                        
                else:
                    filtered_dataset[k] = vs
                    filtered_entities_dataset[k] = entities_dataset_[k]
                    verbose = True
                    output = ' << {:3}/{}: {:>6}/{:<6}, {:.3f} seconds, cohesion:{:.4f}, {} >>'.format(i + 1, 
                                                                                                tot, 
                                                                                                len(entities_dataset_[k]),
                                                                                                initial_length[k],
                                                                                                time.time() - t,
                                                                                                cohesions_[k],
                                                                                                k)
                if verbose:
                    print(output)
                self.log('{}'.format(output))

                t = time.time()
                
            dataset_ = copy.deepcopy(filtered_dataset)
            entities_dataset_ = copy.deepcopy(filtered_entities_dataset)

            output = '{:30}'.format(round(time.time() - while_time, 4))
            print(output)
            self.log(output)
            while_time = time.time()
            
        return dataset_, entities_dataset_, out_words

    def get_stats(self, cohesions):
        '''
        get some stats of the cluster cohesions

        params:
            a dict which have all cohesions on which stats will be computed
        returns:
            the stats
        '''

        v = list(cohesions.values())

        print('SINGLE WORDS CLUSTERS\n')
        print('minimum mean cluster similarities: {}'.format(round(min(v), 4)))
        print('maximum mean cluster similarities: {}'.format(round(max(v), 4)))
        print('mean of mean cluster similarities: {}'.format(round(np.mean(v), 4)))
        print('std  of mean cluster similarities: {}'.format(round(np.std(v), 4)))

        return 'SINGLE WORDS CLUSTERS\n' + \
            'minimum mean cluster similarities: {}\n'.format(round(min(v), 4)) + \
            'maximum mean cluster similarities: {}\n'.format(round(max(v), 4)) + \
            'mean of mean cluster similarities: {}\n'.format(round(np.mean(v), 4)) + \
            'std  of mean cluster similarities: {}'.format(round(np.std(v), 4))
