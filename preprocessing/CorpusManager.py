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
        """
        read the corpus at PATH for the first length lines
        for each line the '\n' is removed, multiple withespaces are compressed to one
        save in the self.corpus a list for each line ('this is a sentence' become ['this', 'is', 'a', 'sentence'])
        save in the self.joined corpus the line
        create the vocab at self.vocab by adding each unique word
        :param 
            PATH: the path to the corpus
            length: the number of lines that are to read
        """
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
         """
        takes a fraction of entities in self.entity_token_in_corpus and for each call the function self.entities_token_in_corpus
        this method (and the involved one) are thought to work in parallel, so after the calling a reduce is applied
        :param 
           n_proc: the number of process used to make the computation
           n_entities: the fraction of entities to find
        """

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
    
    def create_word_occurrence_index(self, word):
        """
        loop on all the corpus, call self.find_in_sentence to find occurrences of word in each sentence, returns a dict
        :param 
            word: the word to find
        :return: a dict with the structure: {entity_name: list of tuples (row: [occurrences in row])}
        """
        key = word
        returning_list = []
        for row_index, sent in enumerate(self.joined_corpus):
            if sent.find(' ' + word + ' ') != -1:
                indices = self.find_in_sentence(word = word, sentence = sent)
                if indices:
                    returning_list.append((row_index, indices))
        return {key: returning_list}

    def find_in_sentence(self, word, sentence):
        """
        returns the indexes in which the word appear in a sentence
        :params
            word: the word to find
            sentence: the sentence in which find the word
        :return: a list of indices
        """

        indices = [i for i, x in enumerate(sentence.split()) if x == word]
        return indices

    def clean_occurrences(self, list_of_indexes):
        return [{k:v} for L in list_of_indexes for k,v in L.items() if v]

    def avoid_multilabeling(self, entity_dict, G, file):
        """
        entity dict is a dict which format is : {concept: [list of entities], ...}, the same entity name can appear in more than one list
        this method remove the repetitions:
            first check which entities are repeated, then for each repeated entities select the concept which is more deeper in the graph,
            if the most deepere concept is not unique, a casual concept (between the deepest) is taken 
        :param 
            entity_dict: the entity dict returned from the method self.entities_from_types
            G: the graph which contains the concepts
            file: path to the log file
        :return: an entity_dict without repeated entities
        """
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
        """
        starting from indexes of useful words in corpus (word_indexes) and the entity to find (entity_dict) 
        returns the occurrences of the entities in entity dict.
        :param 
            word_indexes: the first return of parallel find, contain a couple (row, index) for each word that is in entities names
            entity_dict: the entity dict returned from the method self.entities_from_types
            verbose: if True some (a very high number of) prints are showed
        :return: a dict like word_indexes but with ALL AND ONLY the entities names in corpus, with row indexes and occurrence indexes
        """

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


    def extract_rows(self, word):
        """
        extract the row index from each tuple in word_indexes[word]
        :param
            word: a key in self.word_indexes
        :return: a list of indexes which are the row in which word appear
        """
        return [t[0] for t in self.word_indexes[word]] #extract the row index from each tuple

    def check_entity_in_row(self, ENTITY, ROW, verbose):
        """
        returns the indexes of all occurrences of an entity name (ENTITY) in a sentence (ROW)
        :param 
            ENTITY: an entity name (e.g., 'London', 'New York')
            ROW: a row IN WHICH THE ENTITY OCCURS
            verbose: if True some (a very high number of) prints are showed
        :return: a list of indices which are the occurrence of ENTITY in the row ROW of the corpus
        """

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


    def extract_rows_occurrency(self, word_phrase, row):
        """
        extract the occurrency indexes of each token in the word_phrase for a row in the corpus
        :param
            word_prase: a list of token which compose an entity name (e.g., ['new', 'york'])
            row: a row index of the corpus
        :return: a list of indexes which are occurency indexes of the tokens
        """
        return [t[1] for s in word_phrase for t in self.word_indexes[s] if t[0] == row]