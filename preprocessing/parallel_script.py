from utils import load_data_with_pickle, save_data_with_pickle

PICKLES_PATH = '../../source_files/pickles/'

corpus = load_data_with_pickle(PICKLES_PATH + 'corpus')


def find(word_to_index):
    word_indexes = {}
    line_index = 0
    for sentence in c.corpus[:500]:
        for word_i, word in enumerate(sentence):
            if word in word_to_index:
                try:
                    word_indexes[word].append([line_index, word_i])
                except:
                    word_indexes[word] = [[line_index, word_i]]

        line_index += 1 
    return word_indexes


