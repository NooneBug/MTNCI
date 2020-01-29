
from tqdm import tqdm


class CorpusManager():

    corpus = []
    vocab = set()
    entities_occurrence_index = {}
    concept_entities = {}

    def read_corpus(self, PATH):
        with open(PATH, 'r') as inp:
            ll = inp.readlines()
            for l in tqdm(ll):
                l = l.replace('\n', '')
                l = l.replace('  ', ' ')
                if len(l) > 0:
                    self.corpus.append(l.split(' '))

        self.create_vocab()
    
    def create_vocab(self):
        # USE GENSIM
        """
        create the vocabulary of the corpus 
        :param  :
        :return :
        """
        
        print('building vocab')
        for sentence in tqdm(self.corpus):
            self.vocab = self.vocab.union(set(sentence))

    def check_words(self, entity_dict):
        for concept, entities in tqdm(entity_dict.items()):
            for e in entities:
                if ' ' in e:
                    splitted = e.split(' ')
                    if all(ss in self.vocab for ss in splitted):
                        try:
                            self.concept_entities[concept].append(e)
                        except:
                            self.concept_entities[concept] = [e]
                elif e in self.vocab:
                    try:
                        self.concept_entities[concept].append(e)
                    except:
                        self.concept_entities[concept] = [e]

