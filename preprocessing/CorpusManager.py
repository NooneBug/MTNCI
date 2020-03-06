from multiprocessing import Pool
from tqdm import tqdm
from collections import defaultdict
import re
import time
import random
import copy
import networkx as nx
from colorama import Fore

class CorpusManager():
    corpus = []
    vocab = {}
    word_occurrence_index = {}
    concept_entities_unique = {}
    all_entities_tokens = set()
    joined_corpus = []   

    def __init__(self):
        self.corpus = []
        self.vocab = set()
        self.word_occurrence_index = {}
        self.concept_entities_unique = {}
        self.all_entities_tokens = set()
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
                    splitted = l.split(' ')
                    self.corpus.append(splitted)
                    self.joined_corpus.append(l)
                    for s in splitted:
                        self.vocab.add(s)

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

        self.all_entities_tokens = set()

        for _, entities in entity_dict.items():
            for e in entities:
                splitted = e.split(' ')
                if len(splitted) == 1:
                    self.all_entities_tokens.add(e)
                else:
                    for s in splitted:
                        self.all_entities_tokens.add(s)
        
        self.entities_token_in_corpus = self.all_entities_tokens.intersection(self.vocab)
        self.all_entities_tokens = list(self.all_entities_tokens)

        print('{} words in vocab'.format(len(self.vocab)))
        print('{} entities'.format(len(self.all_entities_tokens)))
        print('{} entity-tokens in vocab'.format(len(self.entities_token_in_corpus)))



        random.shuffle(self.all_entities_tokens)

    def parallel_find(self, n_proc, n_entities = None):
        
        if not n_entities:
            n_entities = 1        

        p = Pool(n_proc)

        print('start indexing')
        frac = int(n_entities * len(self.entities_token_in_corpus))
        es = random.sample(list(self.entities_token_in_corpus), frac)
        
        t = time.time()

        index_list = p.map(self.create_word_occurrence_index, es)

        t = time.time() - t

        index_list_dict = {k:v for elem in index_list for k, v in elem.items() if v}
        p.close()
        return index_list_dict, n_proc, frac, len(self.corpus), t
        
    def find_in_sentence(self, word, sentence):
        indices = [i for i, x in enumerate(sentence.split()) if x == word]
        return indices

    def create_word_occurrence_index(self, word):
        key = word
        returning_list = []
        for row_index, sent in enumerate(self.joined_corpus):
            if sent.find(' ' + word + ' ') != -1:
                indices = self.find_in_sentence(word = word, sentence = sent)
                if indices:
                    returning_list.append((row_index, indices))
        return {key: returning_list}

    def create_word_occurrence_index_set(self, words):
        """
        loop on all the corpus and use re.finditer to search the index and 
        :param 
            e: an entity name
        :return: a dict with the structure: {entity_name: list of tuples (row: [occurrences in row])}
        """

        returning_dict = {}

        for row_index, sent in enumerate(self.joined_corpus):
            inter = set(sent.split(' ')).intersection(words)
            if inter:
                indices = self.find_in_sentence_set(words = inter, sentences = sent)
                for k in indices.keys():
                    try:
                        returning_dict[k].append((row_index, indices[k]))
                    except:
                        returning_dict[k] = [(row_index, indices[k])]
        return returning_dict

    def find_in_sentence_set(self, words, sentences):

        indices = {}
        for i, sentence_word in enumerate(sentences.split()):
            if sentence_word in words:
                # append the word index in sentence
                try:
                    indices[sentence_word].append(i)
                except:
                    indices[sentence_word] = [i]
        return indices

    def clean_occurrences(self, list_of_indexes):
        return [{k:v} for L in list_of_indexes for k,v in L.items() if v]

    def avoid_multilabeling(self, entity_dict, G, file):
        reverse_dict = defaultdict(list)

        for k, words in entity_dict.items():
            for w in words:
                reverse_dict[w].append(k)
        i = 0
        for r in reverse_dict.values():
            if len(set(r)) > 1 :
                i += 1
        
        print('Multilabel alerts: {}'.format(i))

        with open(file, 'w') as out:
            out.write('Multilabel alerts: {}\n'.format(i))

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
                        with open(file, 'a') as out:
                            out.write('remove {} from {}\n'.format(k, clas))
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
        
        with open(file, 'a') as out:
            out.write('Multilabel alerts: {}'.format(i))                
        return entity_dict

    def check_composite_words(self, word_indexes, entity_dict, verbose = False):
        j = 0
        found = defaultdict(list)

        all_entities = []
        self.word_indexes = word_indexes

        for _, v in entity_dict.items():
            all_entities.extend(v)
        all_entities = set(all_entities)

        keys = set(word_indexes.keys())

        bar = tqdm(total = len(all_entities))
        
        for entity in all_entities:
            splitted = entity.split(' ')
            if set(splitted) <= keys:
                # all words in this entity are in the corpus
                i = 0
                rows = self.extract_rows(splitted[i])
                while rows and i + 1 < len(splitted):
                    i += 1
                    rows = [r for r in rows if r in self.extract_rows(splitted[i])]
                if rows and len(splitted) > 1:
                    for r in rows:
                        b = self.check_entity_in_row(ENTITY=entity, ROW=r, verbose = verbose)
                    if b:
                        if verbose:
                            print('entity found at index(es): {}'.format(' '.join([str(v) for v in b])))
                        found[entity].append((r, b))
                    j +=1
                elif rows:
                    found[entity] = word_indexes[entity]
            if not verbose:
                bar.update(1)
        return found
    
    def check_entity_in_row(self, ENTITY, ROW, verbose):
    
        a = self.extract_rows_occurrency(ENTITY.split(' '), ROW)

        if verbose:
            print('row: {}'.format(ROW))
            print('entity: {}'.format(ENTITY))
            print('indexes of each word: {}'.format([(w, o) for w, o in zip(ENTITY.split(' '), a)]))
            print(' '.join([Fore.WHITE + c_w if c_w not in ENTITY.split(' ') else Fore.CYAN + c_w for c_w in self.corpus[ROW]]))
            print(Fore.WHITE + '--------')
        values = []
        for value in a[0]:
            occ = [(value, 0)]
            i = 1
            while i < len(a) and value + i in a[i]:
                occ.append(((value + i), i))
                i += 1
            if i == len(a):
                if verbose:
                    print(Fore.WHITE + 'entity present in this sentence at the index {}'.format(value))
                values.append(value)
            else:
                if verbose:
                    print(Fore.WHITE + 'entity not present in this sentence at the index {}'.format(value))
            if verbose:
                print(Fore.WHITE + '--------')
        if values:
            return values

    def extract_rows(self, word):
        return [t[0] for t in self.word_indexes[word]] #extract the row index from each tuple

    def extract_rows_occurrency(self, word_phrase, row):
        return [t[1] for s in word_phrase for t in self.word_indexes[s] if t[0] == row]

