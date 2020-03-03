from multiprocessing import Pool
from tqdm import tqdm
from collections import defaultdict
import re
import time
import random
import copy
import networkx as nx

class CorpusManager():
    corpus = []
    vocab = {}
    word_occurrence_index = {}
    concept_entities_unique = {}
    all_entities = []
    joined_corpus = []   

    def __init__(self):
        self.corpus = []
        self.vocab = {}
        self.word_occurrence_index = {}
        self.concept_entities_unique = {}
        self.all_entities = []
        self.joined_corpus = []           

    def read_corpus(self, PATH, length = 5000000):
        self.corpus = []
        with open(PATH, 'r') as inp:
            print('read input corpus')
            ll = inp.readlines()
            for l in tqdm(ll[:length]):
                l = l.replace('\n', '')
                l = re.sub(r'[ ]+',' ', l)
                if len(l) > 0:
                    self.corpus.append(l.split(' '))
                    self.joined_corpus.append(' ' + l + ' ')

        # self.create_vocab()
    
    # def create_vocab(self):
    #     """
    #     create the vocabulary of the corpus 
    #     :param  :
    #     :return :
    #     """
    #     print('building vocab')
    #     for sentence in tqdm(self.corpus):
    #         for s in set(sentence):
    #             if len(s) > 2:
    #                 self.vocab[s] = 1
    #     self.vocab = list(self.vocab.keys())

    def create_all_entities(self, entity_dict, concepts = None):
        """
        create a list of all entities names about concepts in the concepts variable
        if concept is None, create all entities names about all concepts
        :param 
            entity_dict: the entity dict returned from the method self.entities_from_types
            concepts: a list of concepts
        """
        if not concepts:
            concepts = list(entity_dict.keys())
        self.all_entities = [w for k, li in entity_dict.items() for w in li if k in concepts]
        random.shuffle(self.all_entities)

    def parallel_find(self, n_proc, n_entities = None, clean = True):
        
        if not n_entities:
            n_entities = len(self.all_entities)        

        t = time.time()
        p = Pool(n_proc)
        print('start indexing')
        # es = copy.deepcopy(random.sample(self.all_entities, n_entities))
        es = copy.deepcopy(self.all_entities[:n_entities])
        # print(es)

        if clean:
            index_list = [x for x in p.map(self.create_word_occurrence_index, es) if x]
        else:
            index_list = [x for x in p.map(self.find_word_in_sentences, es)]

        # rebuild the parallel return

        t = time.time() - t

        index_list_dict = {k:v for elem in index_list for k, v in elem.items() if v}

        return index_list_dict, n_proc, n_entities, len(self.corpus), t


    def find_in_sentence(self, word, sentence):
        indices = [i for i, x in enumerate(sentence.split()) if x == word]
        return indices

    def find_word_in_sentences(self, word):
        returning_list = []
        for row_index, sent in enumerate(self.joined_corpus):
            indices = self.find_in_sentence(word = word, sentence = sent)
            if indices:
                returning_list.append((row_index, indices))
        return {word : returning_list}

    def create_word_occurrence_index(self, word):
        """
        loop on all the corpus and use re.finditer to return the index of all occurrences of the entity e
        :param 
            e: an entity name
        :return: a dict with the structure: {entity_name: list of tuples (row: [occurrences in row])}
        """
        key = word
        returning_list = []

        for row_index, sent in enumerate(self.joined_corpus):
            if sent.find(' ' + word + ' ') != -1:
                indices = self.find_in_sentence(word = word, sentence = sent)
                if indices:
                    returning_list.append((row_index, indices))
                # value = [(i, [m.start() for m in re.finditer(' ' + e + ' ', jo_c)]) ]
        return {key: returning_list}

    def clean_occurrences(self, list_of_indexes):
        return [{k:v} for L in list_of_indexes for k,v in L.items() if v]

    def avoid_multilabeling(self, entity_dict, G):
        reverse_dict = defaultdict(list)

        for k, words in entity_dict.items():
            for w in words:
                reverse_dict[w].append(k)
        i = 0
        for r in reverse_dict.values():
            if len(set(r)) > 1 :
                i += 1

        print('Multilabel alerts: {}'.format(i))

        for k, v in reverse_dict.items():
            if len(set(v)) > 1:
                min_dist = 100
                for clas in v:
                    d = nx.shortest_path_length(G, 'Thing', clas)
                    if d < min_dist:
                        min_dist = d
                        min_k = clas.lower()
                    if d == min_dist:
                        x = random.random()
                        if x > 0.5:
                            min_dist = d
                            min_k = clas.lower()
                for clas in v:
                    if clas != min_k:
                        print('remove {} from {}'.format(k, clas))
                        entity_dict[clas].remove(k)

        reverse_dataset = {}
        for k, words in entity_dict.items():
            for w in words:
                try:
                    reverse_dataset[w].append(k)
                except:
                    reverse_dataset[w] = [k]
        i = 0
        for r in reverse_dataset.values():
            if len(set(r)) > 1:
                i += 1

        print('Multilabel alerts: {}'.format(i))

        return entity_dict

    