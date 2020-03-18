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
        def __init__(self, vector_list, label_list, target_list, device):
            self.X = torch.tensor(vector_list, device = device)
            self.labels = torch.tensor(label_list, device = device)
            self.target_list = {k: torch.tensor(x, dtype = torch.float64, device = device) for k, x in target_list.items()}
            
        def __getitem__(self, index):
            return (self.X[index], 
                    self.labels[index],
                    {k:x[index] for k, x in self.target_list.items()}
                )

        def __len__(self):
            return len(self.X)

class DatasetManager():

    def __init__(self, id):
        self.file_id = id 

    def set_device(self, device):
        self.device = device

    def setup_print_times(self, keywords):
        self.keywords = keywords
        print('-----------------------------')
        print('... loading {} ...'.format(self.keywords))
        self.time = time.time()
    
    def print_loaded(self):
        print('{} loaded in {:.2f} seconds'.format(self.keywords, 
                                                   time.time() - self.time))
        self.time = None
        self.keywords = None


    def load_entities_data(self, X_PATH, Y_PATH, ENTITIES_PATH):
        self.setup_print_times(keywords = 'entities data')
        self.X = load_data_with_pickle(X_PATH)
        self.Y = load_data_with_pickle(Y_PATH)
        self.entities = load_data_with_pickle(ENTITIES_PATH)
        self.print_loaded()

    def load_concept_embeddings(self, CONCEPT_EMBEDDING_PATHS):
        # load concept embeddings

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

        unique_entities = list(set(self.entities))
        direct_labeling = {e: y for e, y in zip(self.entities, self.Y)}
        labels_of_unique_entities = [direct_labeling[e] for e in unique_entities]
        
        counter = Counter(labels_of_unique_entities)
        
        # exclude labels with less than 3 elements, these will not be useful
        print('... excluding labels with less than {} elements ...'.format(exclude_min_threshold))
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
        print('Initial dataset dimension: ')
        print('X: {}, Y : {}, ENTITIES :{}'.format(len(self.X), len(self.Y), len(self.entities)))
        initial_lengt = len(self.Y)
        not_present = []

        for label in list(set(self.Y)):
            try:
                a = self.concept_embeddings['distributional'][label]
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
        Tr = Counter(self.Y_train).most_common()
        Va = Counter(self.Y_val).most_common()
        Te = Counter(self.Y_test).most_common()

        print('{:^30}|{:^30}|{:^30}'.format('Train','Val', 'Test'))
        print('{:-^30}|{:-^30}|{:-^30}'.format('', '', ''))

        for x, y, z in zip(Tr, Va, Te):
            print('{:^25}{:^5}|{:^25}{:^5}|{:^25}{:^5}'.format(x[0], x[1], y[0], y[1], z[0], z[1]))


    def plot_datasets(self):
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
        return [self.numeric_label_map[y] for y in labels]

    def compute_numeric_labels(self):
        self.numeric_label_map = {y: x for x, y in enumerate(set(self.Y_train))}
        self.inverse_numeric_label_map = {v:k for k,v in self.numeric_label_map.items()}

    def create_numeric_dataset(self):
        self.compute_numeric_labels()
        self.Y_numeric_train = self.get_numeric_label(self.Y_train)
        self.Y_numeric_test = self.get_numeric_label(self.Y_test)
        self.Y_numeric_val = self.get_numeric_label(self.Y_val)

    def create_aligned_dataset(self, in_place = True):

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
        self.weights = {}

        for vectors, label in zip([self.Y_numeric_train, self.Y_numeric_val, self.Y_numeric_test],
                                 ['Train', 'Val', 'Test']):
            x_unique = torch.tensor(vectors, device=self.device).unique(sorted=True)
            x_unique_count = torch.stack([(torch.tensor(vectors, device=self.device)==x_u).sum() for x_u in x_unique]).float()
            m = torch.max(x_unique_count)
            labels_weights = m/x_unique_count
            self.weights[label] = labels_weights / torch.norm(labels_weights)
    
    def get_weights(self, label):
        return self.weights[label]


    def normalize(self):
        self.X = normalize(self.X, axis = 1)


    def setup_filter(self, filter_name, log_file_path, filtered_dataset_path, X = None, Y = None, selfXY = True):
        self.selected_filter = Filter().select_filter(filter_name, log_file_path, filtered_dataset_path)
        
        self.selected_filter.set_parameters()

        if selfXY:
            self.selected_filter.set_data(X = self.X, Y = self.Y)
        elif X and Y:
            self.selected_filter.set_data(X = X, Y = Y)
        else:
            raise Exception('Error: pass an X and an Y or set selfXY to true!!!') from e
    
    def filter(self):
        self.selected_filter.filter()


class Filter():

    def __init__(self, log_file_path = None, filtered_dataset_path = None):
        self.FILTERS = {'ClassCohesion': 'CC'}
        self.log_file = log_file_path
        self.filtered_dataset_path = filtered_dataset_path
    
    def set_data(self, X, Y):
        self.X = X
        self.Y = Y

    
    def log(self, text):
        with open(self.log_file, 'a') as out:
            out.write(text + '\n')

    def select_filter(self, filter_name, log_file_path, filtered_dataset_path):
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
        X = [vector for concept, vectors in self.dataset.items() for vector in vectors]
        Y = [concept for concept, vectors in self.dataset.items() for vector in vectors]

        save_data_with_pickle(self.filtered_dataset_path + 'X', X)
        save_data_with_pickle(self.filtered_dataset_path + 'Y', Y)

    def filter(self):

        print('--- FILTERING PROCESS ---')
        print('... creating dataset ...')
        self.create_dataset()

        print('... creating clusters ...')
        self.clusters()

        print('... computing sampled silhouette ...')
        silh = self.sampled_silhouette()

        self.log('silhouette score: {}'.format(silh))

        print('... filtering out quantiles until {} cohesion is reached ... '.format(self.threshold))
        filtered_dataset = self.filter_quantile(threshold = self.threshold, quantile=self.quantile)
        filtered_dataset = {k: vs for k, vs in filtered_dataset.items() if len(vs) > 2}

        print('vectors in filtered dataset:{}'.format(len([v for k, vs in filtered_dataset.items() for v in vs])))
        self.log('vectors in filtered dataset:{}'.format(len([v for k, vs in filtered_dataset.items() for v in vs])))
        
        self.dataset = filtered_dataset

        print('... re-creating clusters ...')
        self.clusters()

        print('... re-computing sampled silhouette ...')
        silh = self.sampled_silhouette()
        self.log('silhouette score on filtered dataset: {}'.format(silh))


        self.save_data()



    def clusters(self):
        self.cluster_sizes, cohesions = self.create_clusters()

        self.cohesions = {k: np.mean(v) for k, v in cohesions.items()}

        self.log(self.get_stats(cohesions = self.cohesions))

    def sampled_silhouette(self):
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

    def set_parameters(self, threshold = 0.6, quantile = 0.05):
        self.threshold = threshold
        self.quantile = quantile

    def create_dataset(self):
        self.dataset = defaultdict(list)

        for x, y in zip(self.X, self.Y):
            self.dataset[y].append(x)
        
        self.dataset = {k:[v for v in values] for k, values in self.dataset.items() if len(values) > 1}


    def create_clusters(self, epsilon = 0.005, iterations = 5, patience_threshold = 10, max_size = 3000):
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

            output = '{:3}/{}: {:10}, {:15} seconds, {}'.format(i + 1, tot, sample_size, round(time.time() - t, 3), k)
            print(output)
            self.log(text = output)
            t = time.time()
            
        print('total_time: {}'.format(round(time.time() - T, 3)))
        self.log(text = 'total_time: {}'.format(round(time.time() - T, 3)))
        
        return cluster_sizes, cohesions

    def filter_quantile(self, threshold, quantile = 0.05):
    
        dataset_ = copy.deepcopy(self.dataset)
        cohesions_ = copy.deepcopy(self.cohesions)
        
        filtered_dataset = {}
        filtered_quantities = {k:0 for k in dataset_}
        tot = len(dataset_)
        initial_length = {k: len(v) for k, v in dataset_.items()}
        
        t = time.time()
        first = True
        filtered_cohesions = [0]
        
        while min(filtered_cohesions) < threshold:
            
            filtered_cohesions = []
            filtered_dataset = {k:[] for k,v in dataset_.items()}
            while_time = time.time()
            
            for i, (k, vs) in enumerate(dataset_.items()):
                if cohesions_[k] < threshold:
                    if self.cluster_sizes[k] < len(vs):
                        class_cluster = sample(vs, self.cluster_sizes[k])
                    else:
                        class_cluster = vs
                    
                    if len(vs) >= 2 and len(class_cluster) >= 2:
                        computed_cohesions = np.mean(cosine_similarity(vs, class_cluster), axis=1)
                        q = np.quantile(computed_cohesions, quantile, interpolation='nearest')

                        computed_cohesion_mask = np.where(computed_cohesions <= q, 0, 1)

                        for j, v in enumerate(vs):
                            if computed_cohesion_mask[j]:
                                filtered_dataset[k].append(v)
                            else:
                                filtered_quantities[k] += 1
                        
                        if len(filtered_dataset[k]) > 2:
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
                            
                            output = '{:3}/{}: {:.2f}, {:.3f} seconds, cohesion:{:.4f}, {}'.format(i + 1, 
                                                                                                tot, 
                                                                                                filtered_quantities[k]/initial_length[k],
                                                                                                time.time() - t,
                                                                                                new_cohesion,
                                                                                                k)
                            

                            
                        else:
                            filtered_dataset[k] = []
                            output = '{:3}/{}: {} has not passed the filter process (no more vectors)'.format(i + 1, 
                                                                                                              tot,
                                                                                                              k)

                    else:
                        filtered_dataset[k] = []
                        output = '{:3}/{}: {} has not passed the filter process (no more vectors)'.format(i + 1, 
                                                                                                          tot,
                                                                                                          k)
                        
                else:
                    filtered_dataset[k] = vs
                    output = '<- {:3}/{}: {:.2f}, {:.3f} seconds, cohesion:{:.4f}, {} ->'.format(i + 1, 
                                                                                                tot, 
                                                                                                filtered_quantities[k]/initial_length[k],
                                                                                                time.time() - t,
                                                                                                cohesions_[k],
                                                                                                k)
                print(output)
                self.log(output)

                t = time.time()
                
            dataset_ = copy.deepcopy(filtered_dataset)
            
            output = '{:30}'.format(round(time.time() - while_time, 4))
            print(output)
            self.log(output)
            while_time = time.time()
            
        return dataset_

    def get_stats(self, cohesions):
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
