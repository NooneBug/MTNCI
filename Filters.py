import time
from abc import ABC, abstractmethod
from preprocessing.utils import save_data_with_pickle, load_data_with_pickle
from collections import Counter, defaultdict
import random
import numpy as np
from random import sample
from sklearn.metrics import silhouette_score
import copy

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
        self.log_file = ''
    
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

    def select_filter(self, filter_name):
        '''
        select the filter based on filter_name

        params:
            filter_name: the name of the filter
        '''
        if filter_name == self.FILTERS['ClassCohesion']:
            return FilterOnClassCohesion()

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
            raise Exception('Error in filtering process, datasets are not aligned') from e

        save_data_with_pickle(self.filtered_dataset_path + 'X', self.X)
        save_data_with_pickle(self.filtered_dataset_path + 'Y', self.Y)
        save_data_with_pickle(self.filtered_dataset_path + 'entities', self.entities)

    def log_out_words(self, out_words):
        '''
        logs the words filtered out by the filtering process

        params:
            out_words: a dict {concept: [list of words]} which will be logged 
        '''
        with open('../source_files/logs/{}_out_words_log.txt'.format(self.threshold), 'w') as out:
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
        with open('../source_files/logs/{}_in_words_log.txt'.format(self.threshold), 'w') as out:
            out.write('\n ------------------------------------------------------ \n')
            for concept, words in sorted(in_words.items()):
                out.write('{}: {} words maintained, {} unique words\n'.format(concept, len(words), len(set(words))))
                c = sorted(Counter(words).items())
                for couple in c:
                    out.write('\t\t {}: {}\n'.format(couple[0], couple[1]))

    def filter_dataset_on_numerosity(self, low_threshold):
        '''
        filters the dataset based on the numerosity of concept vectors, 
            if a concept has less than filtering_threshold vectors will be filtered out
        '''
        self.entities_dataset = {k: vs for k, vs in self.entities_dataset.items() if len(self.dataset[k]) > low_threshold} 
        self.dataset = {k: vs for k, vs in self.dataset.items() if len(vs) > low_threshold}
        
    def filter(self):
        '''
        define the pipeline of the filter
        '''

        filtering_threshold = 100
        
        print('--- FILTERING PROCESS ---')
        print('... creating dataset ...')

        self.create_datasets()
        self.filter_dataset_on_numerosity(filtering_threshold)
        print('... creating clusters ...')
        self.clusters()

        # print('... computing sampled silhouette ...')
        # silh = self.sampled_silhouette()
        # print('silhouette score: {}'.format(silh))
        # self.log('silhouette score: {}'.format(silh))

        print('... filtering out quantiles until {} cohesion is reached ... (quantile = {})'.format(self.threshold, self.quantile))
        t = time.time()
        filtered_dataset, filtered_entities_dataset, out_words = self.filter_quantile(threshold = self.threshold, 
                                                                                      quantile=self.quantile,
                                                                                    filtering_threshold=filtering_threshold)
        print('filtered in {:.2f} seconds'.format(time.time() - t))
        filtered_dataset = {k: vs for k, vs in filtered_dataset.items() if len(vs) > filtering_threshold}
        filtered_entities_dataset = {k: entities for k, entities in filtered_entities_dataset.items() if len(entities) > filtering_threshold}

        self.log_out_words(out_words)
        self.log_in_words(filtered_entities_dataset)

        print('vectors in filtered dataset:{}'.format(len([v for k, vs in filtered_dataset.items() for v in vs])))
        self.log('vectors in filtered dataset:{}'.format(len([v for k, vs in filtered_dataset.items() for v in vs])))
        
        self.dataset = filtered_dataset
        self.entities_dataset = filtered_entities_dataset

        max_number = 4000
        self.reduce_max_number(max_number)

        print('... re-creating clusters ...')
        self.clusters()

        # print('... re-computing sampled silhouette ...')
        # silh = self.sampled_silhouette()
        # print('silhouette score: {}'.format(silh))
        # self.log('silhouette score on filtered dataset: {}'.format(silh))

        # self.save_data()

        X = [vector for concept, vectors in self.dataset.items() for vector in vectors]
        Y = [concept for concept, vectors in self.dataset.items() for vector in vectors]
        entities = [entity for concept, entities in self.entities_dataset.items() for entity in entities]

        return X, Y, entities

    def reduce_max_number(self, max_number):
        counter = {c:len(vectors) for c, vectors in self.dataset.items()}

        support_dict = {}
        new_dataset = {}
        new_entities_dataset = {}
        for c, count in counter.items():
            if count > max_number:
                
                support_list = [(vec, ent) for vec, ent in zip(self.dataset[c], self.entities_dataset[c])]

                sample = random.sample(support_list, max_number)
                new_dataset[c] = [s[0] for s in sample]
                new_entities_dataset[c] = [s[1] for s in sample]

            else:
                new_dataset[c] = self.dataset[c] 
                new_entities_dataset[c] = self.entities_dataset[c]

        self.dataset = new_dataset
        self.entities_dataset = new_entities_dataset

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

    def set_parameters(self, log_file_path, filtered_dataset_path, threshold, quantile, cluster_distance):
        '''
        set the filters parameters

        params:
            log_file_path: the path to the log file
            filtered_dataset_path: the path in which save the the filtered dataset
            threshold: the coherence to be reached by each concept cluster
            quantile: the quantile which will be removed on each iteration until the threshold is not reached 
        '''
        self.log_file = log_file_path
        self.filtered_dataset_path = filtered_dataset_path
        self.threshold = threshold
        self.quantile = quantile
        self.cluster_distance = cluster_distance
        np.random.seed(236451)

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
            patience_threshold: the number of consecutive times in which both mean and standard deviant are lesser than the epsilon value
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
                    
                    cohesion = np.sum(np.tril(self.cluster_distance(sam), -1))/sum([I + 1 for I in range(len(sam) - 1)])
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

    def filter_quantile(self, threshold, filtering_threshold = 2, quantile = 0.05):
        
        '''
        for each concept, create a sample of the cluster size, compute the cohesion of each vector w.r.t the sample
        and iterative exclude all quantiles, re-sample and recompute the cohesion until the threshold is reached 

        params:
            threshold: the cohesion value to be reached from each cluster
            filtering_threshold: the minimum number of value in each cluster
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
        
        first = True
        filtered_cohesions = [0]
        
        all_time = time.time()
        iterations = 0
        while_time = time.time()
        while filtered_cohesions and min(filtered_cohesions) < threshold:
            
            filter_counter = 0
            filtered_cohesions = []
            filtered_dataset = {k:[] for k, _ in dataset_.items()}
            filtered_entities_dataset = {k: [] for k, _ in dataset_.items()}
            t = time.time()
            
            for i, ((k, vs), (_, entities)) in enumerate(zip(dataset_.items(), entities_dataset_.items())):
                if len(vs) != len(entities):
                    raise Exception('ERROR IN VECTOR-ENTITIES ALIGNMENT!!') from e
                if cohesions_[k] < threshold:
                    if self.cluster_sizes[k] < len(vs):
                        class_cluster = sample(vs, self.cluster_sizes[k])
                    else:
                        class_cluster = vs
                    
                    if len(class_cluster) >= filtering_threshold:

                        computed_cohesions = np.mean(self.cluster_distance(vs, class_cluster), axis=1)
                        q = np.quantile(computed_cohesions, quantile, interpolation='nearest')

                        computed_cohesion_mask = np.where(computed_cohesions <= q, 0, 1)

                        for j, (v, e) in enumerate(zip(vs, entities)):
                            if computed_cohesion_mask[j]:
                                filtered_dataset[k].append(v)
                                filtered_entities_dataset[k].append(e)
                            else:
                                filtered_quantities[k] += 1
                                out_words[k].append(e)

                        if len(filtered_dataset[k]) > filtering_threshold:
                            if self.cluster_sizes[k] < len(filtered_dataset[k]):
                                class_cluster = sample(filtered_dataset[k], self.cluster_sizes[k])
                            else:
                                class_cluster = filtered_dataset[k]
                    
                            if len(filtered_dataset[k]) != len(class_cluster):
                                new_cohesion = np.sum(self.cluster_distance(filtered_dataset[k], class_cluster))/(len(filtered_dataset[k] * len(class_cluster)))
                            else:
                                new_cohesion = np.sum(np.tril(self.cluster_distance(filtered_dataset[k]), -1))/sum([I + 1 for I in range(len(filtered_dataset[k]) - 1)])
                                                                            
                            filtered_cohesions.append(new_cohesion)
                            
                            cohesions_[k] = new_cohesion
                            verbose = True
                            filter_counter += 1
                            output = '    {:3}/{}: {:>6}/{:<6}, {:.3f} seconds, cohesion:{:.4f}, {}   '.format(i + 1, 
                                                                                                tot, 
                                                                                                initial_length[k] - filtered_quantities[k],
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
                    verbose = False
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
            iterations += 1
            output = '  {}^ iteration : {:.2f} seconds (total: {:.2f} seconds), {} quantiles filtered out in this iteration'.format(iterations,
                                                                                                                    time.time() - while_time,
                                                                                                                    time.time() - all_time,
                                                                                                                    filter_counter)
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

# class MultilabelFilterOnClassCohesion(FilterOnClassCohesion):
    
    def filter(self):
        '''
        define the pipeline of the filter
        '''

        filtering_threshold = 100
        
        print('--- FILTERING PROCESS ---')
        print('... creating dataset ...')

        self.create_datasets()
        self.filter_dataset_on_numerosity(filtering_threshold)
        print('... creating clusters ...')
        self.clusters()

        # print('... computing sampled silhouette ...')
        # silh = self.sampled_silhouette()
        # print('silhouette score: {}'.format(silh))
        # self.log('silhouette score: {}'.format(silh))

        print('... filtering out quantiles until {} cohesion is reached ... (quantile = {})'.format(self.threshold, self.quantile))
        t = time.time()
        self.filtered_dataset, self.filtered_entities_dataset, self.out_words = self.filter_quantile(threshold = self.threshold, 
                                                                                      quantile=self.quantile,
                                                                                      filtering_threshold=filtering_threshold)

        filtered_dataset, filtered_entities_dataset, out_words_2 = self.multilabel_quantile_filter(threshold = self.threshold, 
                                                                                                    quantile=self.quantile,
                                                                                                    filtering_threshold=filtering_threshold)

        print('filtered in {:.2f} seconds'.format(time.time() - t))
        filtered_dataset = {k: vs for k, vs in filtered_dataset.items() if len(vs) > filtering_threshold}
        filtered_entities_dataset = {k: entities for k, entities in filtered_entities_dataset.items() if len(entities) > filtering_threshold}

        self.log_out_words(self.out_words)
        self.log_in_words(filtered_entities_dataset)

        print('vectors in filtered dataset:{}'.format(len([v for k, vs in filtered_dataset.items() for v in vs])))
        self.log('vectors in filtered dataset:{}'.format(len([v for k, vs in filtered_dataset.items() for v in vs])))
        
        self.dataset = filtered_dataset
        self.entities_dataset = filtered_entities_dataset

        max_number = 4000
        self.reduce_max_number(max_number)

        print('... re-creating clusters ...')
        self.clusters()

        # print('... re-computing sampled silhouette ...')
        # silh = self.sampled_silhouette()
        # print('silhouette score: {}'.format(silh))
        # self.log('silhouette score on filtered dataset: {}'.format(silh))

        # self.save_data()

        X = [vector for concept, vectors in self.dataset.items() for vector in vectors]
        Y = [concept for concept, vectors in self.dataset.items() for vector in vectors]
        entities = [entity for concept, entities in self.entities_dataset.items() for entity in entities]

        return X, Y, entities

    def create_datasets(self):
        '''
        create the dataset which will be used in all the pipeline
        '''
        self.multilabel_dataset = defaultdict(list)
        self.multilabel_entities_dataset = defaultdict(list)

        self.dataset = defaultdict(list)
        self.entities_dataset = defaultdict(list)


        pattern = r'_[a-z]+'
        
        all_count = Counter(self.entities)

        all_entities_without_labels = [re.sub(pattern, '', e) for e in set(self.entities)]

        single_count = Counter(all_entities_without_labels)

        for x, y, e in zip(self.X, self.Y, self.entities):
            cleaned_entity = re.sub(pattern, '', e)
            if single_count[cleaned_entity] == all_count[e]:
                self.dataset[y].append(x)
                self.entities_dataset[y].append(e) 
            else:
                self.multilabel_dataset[y].append(x)
                self.multilabel_entities_dataset[y].append(e)
            
        
        self.dataset = {k:values for k, values in self.dataset.items() if len(values) > 1}
        self.entities_dataset = {k:entities for k, entities in self.entities_dataset.items() if len(entities) > 1}

        self.multilabel_dataset = {k:values for k, values in self.single_label_dataset.items() if len(values) > 1}
        self.multilabel_entities_dataset = {k:entities for k, entities in self.single_label_entities_dataset.items() if len(entities) > 1}

    def multilabel_quantile_filter(self, threshold, quantile, filtering_threshold, sub = 10):
        out_words = defaultdict(list)
        for concept, vectors in self.multilabel_dataset.items():
            entities = self.multilabel_entities_dataset[concept]

            similarities = np.mean(self.cluster_distance(vectors, random.sample(self.filtered_dataset[y], self.cluster_sizes[y])), axis=1)

            couples = sorted([(x, s, e) for x, s in zip(vectors, similarities, entities)], key=lambda x: x[1], reverse=True)

            for i in range(len(couples), sub):
                batch = [x[0] for x in couples[i : i + sub]]
                entities_batch = [x[2] for x in couples[i : i + sub]]
                
                new_cluster = []
                new_cluster.extend(filtered_dataset[y])
                new_cluster.extend(batch)

                cohesion = np.sum(np.tril(self.cluster_distance(new_cluster), -1))/sum([I + 1 for I in range(len(new_cluster) - 1)])

                if cohesion > threshold:
                    self.filtered_dataset[y] = new_cluster
                    self.filtered_entities_dataset[y].extend
                else:
                    out_words[y] = list(set([x[2] for x in couples[i:]]))
                    break
        return self.filtered_dataset, self.filtered_entities_dataset, out_words
