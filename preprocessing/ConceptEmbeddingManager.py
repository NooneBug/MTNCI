from utils import load_data_with_pickle, save_data_with_pickle
import networkx as nx
from operator import itemgetter
from collections import defaultdict
from tqdm import tqdm
import random
import os
import numpy as np
from gensim.models import Word2Vec

DATA_PATH = '../hyperbolics/data/edges/'

EMBEDDING_PATH = '../../source_files/embeddings/'

class ConceptEmbedding():

    def __init__(self):
        self.HYPER_E = 'HyperE'
        self.TYPE2VEC = 'Type2Vec'

        # self.transitive_closure = nx.algorithms.dag.transitive_closure(self.tree)
    def read_tree(self, tree_path):
        self.tree = load_data_with_pickle(tree_path)

    def set_model_name(self, model_name):
        self.model_name = model_name

    def create_embedding(self, embedding_name):
        print('building the {} concept embedding',format(self.model_name))
        self.save_path = EMBEDDING_PATH + embedding_name

class HyperEEmbeddingManager(ConceptEmbedding):
    
    def _init__(self, tree_path):
        super()


    def create_dictionary(self):
        self.dic = {}
        i = 0

        for n in list(nx.traversal.bfs_tree(self.tree, 'Thing')):
            self.dic[n] = i
            i += 1
        
        self.inv_dic = {str(v): k for k, v in self.dic.items()}


    def create_numeric_edgelist(self):
        ordered_edgelist = {n:list(self.tree.successors(n)) for n in self.tree.nodes()}
        edgelist = defaultdict(list)
        for k, v in ordered_edgelist.items():
            for al in v:
                edgelist[k].append(al)
        
        ordered_edgelist = sorted(edgelist.items(), key=itemgetter(0))
        
        with open(DATA_PATH + 'MTNCI_edgelist', 'w') as out:
            for s, v in ordered_edgelist:
                for t in v:
                    out.write('{} {}\n'.format(self.dic[s], self.dic[t]))

    def create_embedding(self, embedding_name):
        super()
        
        self.create_dictionary()
        self.create_numeric_edgelist()

        os.chdir('../hyperbolics/Docker')
        os.system('docker build -t hyperbolics/gpu .')
        os.chdir('..')
        os.system('nvidia-docker run -v "$PWD:/root/hyperbolics" -it hyperbolics/gpu julia combinatorial/comb.jl -d data/edges/MTNCI_edgelist -m {} -e 1.0 -p 64 -r 2 -a -s'.format(embedding_name))
        print(os.getcwd())
        os.system('mv {} ../../source_files/embeddings/'.format(embedding_name))

        self.embedding = self.import_stanford_hyperbolic(PATH = self.save_path)
        print('------------------')
        print(self.embedding)

    def import_stanford_hyperbolic(self, PATH):
        with open(PATH, 'r') as inp:
            lines = inp.readlines()
            lines = lines[1:]
            lines = [l.replace('\n','') for l in lines]
            tau = float(lines[0].split(',')[-1])
            stanford_emb = {self.inv_dic[l.split(',')[0]]: np.array(l.split(',')[1:-1]).astype('float64') for l in lines}
        return stanford_emb, tau


class Type2VecEmbeddingManager(ConceptEmbedding):
    def __init__(self, concept_corpus_path):
        super()
        self.concept_corpus_path = concept_corpus_path

    def create_embedding(self, embedding_name, remove_mode):
        super()
        self.save_path = EMBEDDING_PATH + embedding_name
        self.remove_mode = remove_mode
        self.cleaned_corpus = self.clean_corpus(known = list(self.tree.nodes()))
        self.embedding = self.train_t2v()
        print('------')
        print(self.embedding)

    def clean_corpus(self, known):
        removed = 0
        total = 0

        cleaned_type_corpus = []

        with open(self.concept_corpus_path) as f:
            content = f.readlines()
            sentences = [x.strip() for x in content]
            for s in tqdm(sentences):
                cleaned_sentence = []
                words = s.split(' ')
                for w in words:
                    if w == 'owl#Thing':
                        cleaned_sentence.append('Thing')
                    elif w in known:
                        cleaned_sentence.append(w)
                    else:
                        replacement = self.remove_with_criteria(w)
                        if replacement:
                            cleaned_sentence.append(replacement)
                        removed += 1
                    total += 1
                if cleaned_sentence: 
                    cleaned_type_corpus.append(cleaned_sentence)

        print('removed {} word on {}, ({:.2f}%)'.format(removed, total, removed/total))
        return cleaned_type_corpus


    def remove_with_criteria(self, concept_to_remove):
        if self.remove_mode == 'Remove':
            pass
        elif self.remove_mode == 'Father':
            return list(self.tree.predecessors(concept_to_remove))[0]
        elif self.remove_mode == 'Ancestor':
            return random.choice(list(nx.algorithms.dag.ancestors(self.tree, concept_to_remove)))
        else:
            print('Error, no remove_mode selected')        

    def train_t2v(self):
        model = Word2Vec(self.cleaned_corpus, 
                        workers=24,
                        size = 10
                        )
        model.save(self.save_path)
        return model

    
        