


class ConceptInventionManager():


    def set_dataset_manager(self, datasetManager):
        self.datasetManager = datasetManager
    

    def set_excluding_routing(self, excluding_routine_name):


    def exclude_invention_class(self):


class ExcludingRoutine():

    def exclude(self, *args, **kwargs):
        print('please, choose an excluding routine')

    def set_data(self, X, Y, entities):
        pass

class SlaveExcluder(ExcludingRoutine):

    def exclude(self, concept_to_exclude, splitted = True):
        if concept_to_exclude in set(self.datasetManager.Y):
            if splitted:
                self.xcluded_train_vectors = [v for v, y in zip(self.dataset)]
            else:



