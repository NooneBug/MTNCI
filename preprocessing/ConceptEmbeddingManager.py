from utils import load_data_with_pickle, save_data_with_pickle
import networkx as nx
from operator import itemgetter
from collections import defaultdict
from tqdm import tqdm
import random
import os
import numpy as np
from gensim.models import Word2Vec

# Set of classes to deploy models that can be used to represent concept embedding
# Deployed models: HyperE ( https://github.com/HazyResearch/hyperbolics )
#                  Type2Vec ( https://github.com/vinid/type2vec ) 


# PATH in which HyperE will find the numeric edgelist
DATA_PATH = '../hyperbolics/data/edges/'

# PATH in which Concept Embeddings will be stored
EMBEDDING_PATH = '../../source_files/embeddings/'

class ConceptEmbedding():

    def __init__(self):
        '''
        Setup the constant useful to drive the behaviour
        '''
        self.HYPER_E = 'HyperE'
        self.TYPE2VEC = 'Type2Vec'

        # self.transitive_closure = nx.algorithms.dag.transitive_closure(self.tree)
    def read_trees(self, tree_path, pruned_tree_path):
        '''
        load two networkx graph (must be a tree)
        params:
            tree_path: path to a pickle which contain a networkx tree
            pruned_tree_path: path to a pickle which contain a networkx tree
        '''
        self.tree = load_data_with_pickle(tree_path)
        self.pruned_tree = load_data_with_pickle(pruned_tree_path)

    def set_model_name(self, model_name):
        ''' 
        setup the model name
        params:
            model_name: possible values:
                            self.HYPER_E ( https://github.com/HazyResearch/hyperbolics )
                            self.TYPE2VEC ( https://github.com/vinid/type2vec ) 
        '''
        self.model_name = model_name

    def create_embedding(self, embedding_name):
        '''
        setup the embedding creation, needs to be reimplemented in each class
        :params
            embedding_name: the relative path in which the embedding will be saved, will be combined with EMBEDDING_PATH
        '''
        print('building the {} concept embedding',format(self.model_name))
        self.save_path = EMBEDDING_PATH + embedding_name

class HyperEEmbeddingManager(ConceptEmbedding):
    
    # Class which deploys HyperE ( https://github.com/HazyResearch/hyperbolics )
    
    def _init__(self, tree_path):
        super()

    def create_dictionary(self):
        '''
        create a dictionary which is {node: value} and the inverse dictionary which is {value: node}
        this is because HyperE wants an edgelist in which node names are number, (with the root which is the O (zero) value)
        '''
        self.dic = {}
        i = 0

        for n in list(nx.traversal.bfs_tree(self.pruned_tree, 'Thing')):
            self.dic[n] = i
            i += 1
        
        self.inv_dic = {str(v): k for k, v in self.dic.items()}


    def create_numeric_edgelist(self):
        '''
        create and save at DATA_PATH + 'MTNCI_edgelist' the numeric edgelist
        '''
        ordered_edgelist = {n:list(self.pruned_tree.successors(n)) for n in self.pruned_tree.nodes()}
        edgelist = defaultdict(list)
        for k, v in ordered_edgelist.items():
            for al in v:
                edgelist[k].append(al)
        
        ordered_edgelist = sorted(edgelist.items(), key=itemgetter(0))
        
        with open(DATA_PATH + 'MTNCI_edgelist', 'w') as out:
            for s, v in ordered_edgelist:
                for t in v:
                    out.write('{} {}\n'.format(self.dic[s], self.dic[t]))

    def create_embedding(self, embedding_name, dimensions = 2):
        
        '''
        combines docker and shell commands to build and deploy HyperE
        :params
            embedding_name: the name of the produced embedding
            dimensions: the dimensions used to make the embedding
        '''
        
        super()
        
        if type(dimension) != int:
            raise Exception('Dimensions needs to be a number!!')
        
        self.create_dictionary()
        self.create_numeric_edgelist()

        os.chdir('../hyperbolics/Docker')
        os.system('docker build -t hyperbolics/gpu .')
        os.chdir('..')
        os.system('nvidia-docker run -v "$PWD:/root/hyperbolics" -it hyperbolics/gpu julia combinatorial/comb.jl -d data/edges/MTNCI_edgelist -m {} -e 1.0 -p 64 -r {} -a -s'.format(embedding_name, dimensions))
        print(os.getcwd())
        os.system('mv {} ../../source_files/embeddings/'.format(embedding_name))

        self.embedding = self.import_stanford_hyperbolic(PATH = self.save_path)
        print('------------------')
        print(self.embedding)

    def import_stanford_hyperbolic(self, PATH):
        '''
        parse the julia script output and build a dictionary which become the embedding
        :params
            PATH: the path in which the output of HyperE is stored
        : return
            stanford_emb: a dictionary in format {key: vector}, in this project keys are the concepts
            tau: the tau value which is the length of an arc, i.e. the distance between nodes which share an edge
        '''
        
        with open(PATH, 'r') as inp:
            lines = inp.readlines()
            lines = lines[1:]
            lines = [l.replace('\n','') for l in lines]
            tau = float(lines[0].split(',')[-1])
            stanford_emb = {self.inv_dic[l.split(',')[0]]: np.array(l.split(',')[1:-1]).astype('float64') for l in lines}
        return stanford_emb, tau


class Type2VecEmbeddingManager(ConceptEmbedding):
    
    # Class which deploys Type2Vec ( https://github.com/vinid/type2vec ) 
    
    def __init__(self, concept_corpus_path):
        super()
        self.concept_corpus_path = concept_corpus_path

    def create_embedding(self, embedding_name, remove_mode):
        
        '''
        build the embedding according to the type2vec project
        :params
            embedding_name: the name of the produced embedding
            remove_mode: the modality according to which nodes that are not in tree (but are in the corpus) are replaced (see  self.remove_with_critera)
        '''
        
        super()
        self.save_path = EMBEDDING_PATH + embedding_name
        self.remove_mode = remove_mode
        self.cleaned_corpus = self.clean_corpus(known = list(self.pruned_tree.nodes()))
        self.embedding = self.train_t2v()

    def clean_corpus(self, known):
        '''
        remove from the concept corpus all tokens (concepts) which are not in known
        the remove is made by self.remove_with_criteria which decide how replace the removed tokens
        :params
            known: a list of concept which will NOT be removed from the corpus
        :returns
            cleaned_type_corpus: a corpus composed ONLY of tokens in known
        '''
        
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
                        replacement = -1
                        while replacement not in knwon and replacement != '':
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
        '''
        driven by the remove mode, replace the concept_to_remove with a new one (or don't replace it)
        
        Replacements modes: 'Remove': simply remove the concept_to_remove
                            'Father': substitute the concept_to_remove with his father
                            'Ancestor': substitute the concept_to_remove with a random ancestor
        
        :params
            concept_to_remove: the concept which will be removed
        '''
        if self.remove_mode == 'Remove':
            pass
        elif self.remove_mode == 'Father':
            return list(self.tree.predecessors(concept_to_remove))[0]
        elif self.remove_mode == 'Ancestor':
            return random.choice(list(nx.algorithms.dag.ancestors(self.tree, concept_to_remove)))
        else:
            print('Error, no remove_mode selected')        

    def train_t2v(self):
        '''
        train a Word2Vec model, save it and returns it
        return:
            model: a gensim Word2Vec model
        '''
        model = Word2Vec(self.cleaned_corpus, 
                        workers=24,
                        size = 10
                        )
        model.save(self.save_path)
        return model

    
        
