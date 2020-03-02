from multiprocessing import Pool
from tqdm import tqdm
from collections import defaultdict
import re
import time

class CorpusManager():
    corpus = []
    vocab = {}
    word_occurrence_index = {}
    concept_entities_unique = {}
    all_entities = []
    joined_corpus = []   


    def read_corpus(self, PATH):
        with open(PATH, 'r') as inp:
            ll = inp.readlines()
            print('read input corpus')
            for l in tqdm(ll):
                l = l.replace('\n', '')
                l = l.replace('  ', ' ')
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

    def create_all_entities(self, entity_dict):
        self.all_entities = [w for li in entity_dict.values() for w in li]

    def parallel_find(self, n_proc, n_entities = None):
        
        if not n_entities:
            n_entities = len(self.all_entities)        
        # offset = 10000

        # self.pbar = tqdm(total=5)

        p = Pool(n_proc)
        t = time.time()
        index_list = p.imap(self.create_word_occurrence_index, self.all_entities[0:50])
        print('{} processes, 5000 entities, {:.2f} seconds'.format(n_proc, time.time() - t))
        return index_list

    def create_word_occurrence_index(self, e):
        # self.pbar.update(1)
        return {e: [(i, [m.start() for m in re.finditer(e, jo_c)]) for i, jo_c in enumerate(self.joined_corpus) if jo_c.find(e) != -1]}
        
        # words_to_search = set([e for k, es in self.concept_entities_unique.items() for e in es[0]])
        
        # for sentence in tqdm(self.corpus[:500]):
        #     common = words_to_search.intersection(sentence)
        #     indexes = [(x, i) for x in common for i, s in enumerate(sentence) if s == x]
        #     self.word_occurrence_index.append(indexes)

    def clean_occurrences(self, list_of_indexes):
        return [{k:v} for k,v in list_of_indexes.items() if v]

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


