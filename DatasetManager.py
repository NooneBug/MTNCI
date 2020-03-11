from sklearn.model_selection import train_test_split
from collections import Counter

class DatasetManager():

    def split_data_by_unique_entities(self, X, Y, entities):

        unique_entities = list(set(entities))
        dicrect_labeling = {e: y for e, y in zip(entities, Y)}

        labels_of_unique_entities = [dicrect_labeling[e] for e in unique_entities]

        entities_train, entities_test, \
        y_train_split, y_test_split = train_test_split(unique_entities,
                                                       labels_of_unique_entities,
                                                       test_size = 0.1,
                                                       stratify = labels_of_unique_entities)

        entities_train, entities_val, \
        y_train, y_val = train_test_split(entities_train,
                                          y_train_split,
                                          test_size = 0.1,
                                          stratify = y_train_split)
   
        self.X_train = []
        self.Y_train = []
        self.E_train = []

        self.X_val = []
        self.Y_val = []
        self.E_val = []

        self.X_test = []
        self.Y_test = []
        self.E_test = []

        for x, y, e in zip(X, Y, entities):
            if e in entities_train:
                self.X_train.append(x)
                self.Y_train.append(y)        
                self.E_train.append(e)
                
            elif e in entities_test:
                self.X_test.append(x)
                self.Y_test.append(y)
                self.E_test.append(e)
                
            elif e in entities_val:
                self.X_val.append(x)
                self.Y_val.append(y)
                self.E_val.append(e)
            else:
                print('ERROR IN DATASET SPLITTING')

    def print_statistic_on_dataset(self):
        Tr = Counter(self.Y_train).most_common()
        Va = Counter(self.Y_val).most_common()
        Te = Counter(self.Y_test).most_common()

        print('{:^30}|{:^30}|{:^30}'.format('Train','Val', 'Test'))
        print('{:-^30}|{:-^30}|{:-^30}'.format('', '', ''))

        for x, y, z in zip(Tr, Va, Te):
            print('{:^25}{:^5}|{:^25}{:^5}|{:^25}{:^5}'.format(x[0], x[1], y[0], y[1], z[0], z[1]))
