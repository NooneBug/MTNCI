from SPARQLWrapper import SPARQLWrapper, JSON
import gensim
from string import ascii_letters
import re
from tqdm import tqdm


SIMPLE = 'simple'
VRANK = 'vrank'
COUNT = 'count'

class EntityNameRetriever():

    def __init__(self, query=None):
        self.classtype = ""
        self.setQuery()
        
    def setQuery(self, query_type ='vrank', classtype = None,  query = None):
        """
        setup a the query 
        :param  
            query_type: 
                - 'count' : setup a query which counts the number of entities with type == classtype
                - 'simple': setup a query which returns the entities with type == classtype
                - 'vrank' : setup a query which returns the entities with type == classtype and which has a vrank, ordered by vrank
                - query-like string    : setup the query given as parameter
            classtype : the ontological type which query will ask for entities (sorry for the english)
            query : a query that is executed if query_type = None
        :return:  
        """
        if classtype:
            self.set_classtype(classtype)

        self.query_type = query_type

        if query_type == VRANK:
            self.query = """
                            PREFIX rdf:<http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                            PREFIX dbo:<http://dbpedia.org/ontology/>
                            PREFIX vrank:<http://purl.org/voc/vrank#>
                            SELECT distinct ?s ?v
                            FROM <http://dbpedia.org>
                            FROM <http://people.aifb.kit.edu/ath/#DBpedia_PageRank>
                            WHERE { ?s rdf:type dbo:""" + self.classtype + """.
                                    ?s vrank:hasRank/vrank:rankValue ?v.
                                } 
                            ORDER BY DESC(?v)
                        """

        elif query_type == SIMPLE:
            self.query = """
                            PREFIX rdf:<http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                            PREFIX dbo:<http://dbpedia.org/ontology/>
                            SELECT distinct ?s
                            FROM <http://dbpedia.org>
                            WHERE { ?s rdf:type dbo:""" + self.classtype + """.
                                } 
                        """

        elif query_type == COUNT:
            self.query = """
                            PREFIX rdf:<http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                            PREFIX dbo:<http://dbpedia.org/ontology/>
                            SELECT distinct COUNT(*)
                            FROM <http://dbpedia.org>
                            FROM <http://people.aifb.kit.edu/ath/#DBpedia_PageRank>
                            WHERE { ?s rdf:type dbo:""" + self.classtype + """.
                                } 
                        """
        else:
            self.query = query

    def set_classtype(self, classtype):
        self.classtype = classtype

    def sparqlQuery(self):

        """
        setup sparqlWrapper, set the classtype and launch a query which retrieves entities
        which are described as <entity, a, classtype> in dbpedia, 
        :param 
        :return: a list with entities URIs 
        """

        sparql = SPARQLWrapper("http://dbpedia.org/sparql")

        sparql.setQuery(self.query)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()

        return self.retrieve_results(results)

    def retrieve_results(self, results):

        """
        parse the result of SPARQLWrapper query

        :param  
            results: result returned from SPARQLWrapper

        :return: value if the result is from a count, list otherwise 
        """


        b = results['results']['bindings']
        if self.query_type != 'count':
            collector = []

            for result in b:
                data_a = result["s"]["value"].replace("http://dbpedia.org/resource/", "")
                collector.append((data_a))
            return collector
        elif self.query_type == 'count':
            return int(b[0]['callret-0']['value'])


    def entities_from_types(self, typelist):
        """
        check how much entities can be retrieved, 
            if the number is higher than 10k (dbpedia limit) calls a query that returns 
                the first 10k entities ranked with pagerank, 
            elsewhere returns the retrieved entities
        entities are retrieved via another query

        :param  
            typelist: a list of dbpedia classes (e.g., ['dbo:City', 'dbo:Animal'])
        :return: dict which is formatted as: {type: list_of_entities} 
        """

        entity_dict = {}
        for classtype in tqdm(typelist):
            e = EntityNameRetriever()
            e.setQuery(query_type = COUNT, classtype = classtype)
            count = e.sparqlQuery()
            if count > 0:
                if count >= 10000:
                    e.setQuery(query_type = VRANK, classtype = classtype)
                    entity_list = e.sparqlQuery()
                else:
                    e.setQuery(query_type = SIMPLE, classtype = classtype)
                    entity_list = e.sparqlQuery()
            else:
                entity_list = []
            entity_dict[classtype] = entity_list

        return entity_dict

    def entity_name_preprocessing(self, entity_dict, max_words = 20):
        """
        clean the entity names
        :param 
            entity_dict: the entity dict returned from the method self.entities_from_types
            max_words: number of maximum words allowed, 
                i.e., entity names composed of more than max_words are filtered out
        :return: a dict with the same format of the entity_dict but with entity names cleaned
        """

        cleaned_entity_dict = {k:[] for k in entity_dict.keys()}

        for k, values in entity_dict.items():
            for v in values:
                v = self.preprocess(v)
                if max_words and len(v.split(' ')) <= max_words and v != '':
                    cleaned_entity_dict[k].append(v)
        return cleaned_entity_dict

        
    def preprocess(self, stri):

        """
        define a pipeline which clean a string, 
        :param
            stri: the string to be cleaned 
        :return: the cleaned string
        """

        stri = stri.lower()
        stri = gensim.utils.deaccent(stri)
        stri = ' '.join(stri.split('-'))
        stri = ' '.join(stri.split('_'))
        stri = stri.replace('\'s', '')
        stri = stri.replace('\'', '')
        stri = re.sub(r'[^\w\s]','', stri)
        stri = re.sub(r'[0-9]+',' ', stri)
        stri = re.sub(r'[ ]+',' ', stri)
        stri = stri.strip()

        return stri




