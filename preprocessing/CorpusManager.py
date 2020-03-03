from multiprocessing import Pool
from tqdm import tqdm
from collections import defaultdict
import re
import time
import random
import copy

class CorpusManager():
    corpus = []
    vocab = {}
    word_occurrence_index = {}
    concept_entities_unique = {}
    all_entities = []
    joined_corpus = []   


    def read_corpus(self, PATH, length = 5000000):
        with open(PATH, 'r') as inp:
            print('read input corpus')
            ll = inp.readlines()
            for l in tqdm(ll[:length]):
                l = l.replace('\n', '')
                l = re.sub(r'[ ]+',' ', l)
                if len(l) > 0:
                    self.corpus.append(l.split(' '))
                    self.joined_corpus.append(l)

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
        # offset = 10000

        # self.pbar = tqdm(total=5)

        t = time.time()
        p = Pool(n_proc)
        print('start indexing')
        es = copy.deepcopy(self.all_entities[0:n_entities])
        if clean:
            index_list = [x for x in p.imap(self.create_word_occurrence_index, es) if x]
        else:
            index_list = [x for x in p.imap(self.create_word_occurrence_index, es)]

        return index_list, n_proc, n_entities, len(self.corpus), t

    def create_word_occurrence_index(self, e):
        """
        loop on all the corpus and use re.finditer to return the index of all occurrences of the entity e
        :param 
            e: an entity name
        :return: a dict with the structure: {entity_name: list of tuples (row: [occurrences in row])}
        """
        key = e
        value = [(i, [m.start() for m in re.finditer(e, jo_c)]) for i, jo_c in enumerate(self.joined_corpus) if jo_c.find(e) != -1]
        if value:
            return {key: value}

    def clean_occurrences(self, list_of_indexes):
        return [{k:v} for L in list_of_indexes for k,v in L.items() if v]

    # def check_words(self, entity_dict):
        # for concept, entities in tqdm(entity_dict.items()):

            # single = [e for e in entities if len(e.split(' ')) == 1]
            # phrases = [e for e in entities if len(e.split(' ')) > 1]
            # phrases = [s for e in phrases for s in e.split(' ')]
            
            # single.extend(phrases)

            # self.concept_entities_unique[concept] = [set(single).intersection(set(self.vocab))]
            
            # for e in tqdm(entities):
            #     if ' ' in e:
            #         splitted = e.split(' ')
            #         if all(ss in self.vocab for ss in splitted):
            #             try:
            #                 self.concept_entities[concept].append(e)
            #             except:
            #                 self.concept_entities[concept] = [e]
            #     elif e in self.vocab:
            #         try:
            #             self.concept_entities[concept].append(e)
            #         except:
            #             self.concept_entities[concept] = [e]


    # def create_word_to_index(self):
    #     self.word_to_index = []
    #     for k, sublist in self.concept_entities.items():
    #         for v in sublist:
    #             if ' ' in v:
    #                 for sp in v.split(' '):
    #                     self.word_to_index.append(sp)
    #             else:
    #                 self.word_to_index.append(v)
                

    # def parallel_find(self, n_proc):
        

    #     def divide_chunks(l, n): 
    #         for i in range(0, len(l), n):
    #             yield [(i+j, line) for j, line in enumerate(l[i:i + n])]
        
     
    #     p = Pool(n_proc)
        
    #     indexes_and_corpus_slices = divide_chunks(self.corpus[:500], n_proc)

    #     print('corpus sliced')

    #     word_indexes = defaultdict(list)

    #     for dictionary in p.imap_unordered(self.find, indexes_and_corpus_slices):
    #         for k, v in dictionary.items():
    #             word_indexes[k].extend(v)    

    #     print(word_indexes)

        # return p.map(self.find, indexes_and_corpus_slices)

    # def find(self, corpus):
    #     w_i = {}
    #     for K, sentence in corpus:
    #         for word_i, word in enumerate(sentence):
    #             if word in self.word_to_index:
    #                 try:
    #                     w_i[word].append([K, word_i])
    #                 except:
    #                     w_i[word] = [[K, word_i]]
    #     return w_i  


