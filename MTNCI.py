from torch import nn
import torch
import geoopt
from geooptModules import MobiusLinear, mobius_linear, create_ball
from torch.utils.data import Dataset, DataLoader
from geoopt.optim import RiemannianAdam
from tensorBoardManager import TensorBoardManager
from abc import ABC, abstractmethod
from preprocessing.utils import hyper_distance, hyperbolic_midpoint, cosine_dissimilarity, vector_mean
import numpy as np



class CommonLayer(nn.Module):
    def __init__(self, 
                input_d,
                dims = None,
                dropout_prob = 0):

        super().__init__()
        
        prec = input_d
        self.fully = nn.ModuleList()
        self.bns = nn.ModuleList()

        for dim in dims:
            self.fully.append(nn.Linear(prec, dim).cuda())
            self.bns.append(nn.BatchNorm1d(dim).cuda())
            prec = dim            
        
        self.dropout = nn.Dropout(p=dropout_prob).cuda()
        self.leaky_relu = nn.LeakyReLU(0.1).cuda()
        
    def forward(self, x):
        for i in range(len(self.fully)):
            x = x.double()
            x = self.dropout(self.bns[i](self.leaky_relu(self.fully[i](x))))
        return x

class RegressionOutput(nn.Module):
    def __init__(self, hidden_dim, dims, manifold):
        super().__init__()
        self.out = nn.ModuleList()
        
        self.dropout = nn.Dropout(p=0.1).cuda()
        self.leaky_relu = nn.ReLU().cuda()
        self.bns = nn.ModuleList()
        
        prec = hidden_dim
        
        for dim in dims:
            if manifold == 'poincare':
                self.out.append(MobiusLinear(prec, dim).cuda())
            elif manifold == 'euclid':
                self.out.append(nn.Linear(prec, dim).cuda())
            else:
                print('ERROR: NO MODE SELECTED')
                
            self.bns.append(nn.BatchNorm1d(dim).cuda())
            prec = dim
            
    def forward(self, x):
        for i in range(len(self.out) - 1):
            x = self.leaky_relu(self.out[i](x))
        out = self.out[-1](x)
        return out


class MTNCI(nn.Module):    
    def __init__(self,
                 input_d, 
                 dims = None,
                 out_spec = [{'manifold':'euclid', 'dim':[10]},
                             {'manifold':'poincare', 'dim':[2]}],
                 dropout_prob = 0.2, 
                ):
        
        super().__init__()
        self.common_network = CommonLayer(input_d=input_d,
                                        dims = dims,
                                        dropout_prob=dropout_prob)
        
        self.out_layers = nn.ModuleList()
        
        for spec in out_spec:
            self.out_layers.append(RegressionOutput(hidden_dim=dims[-1],
                                            dims=spec['dim'],
                                            manifold = spec['manifold']))

        

    
        
    def forward(self, x):
        x = self.common_network(x)
        
        output = []
        
        for layer in self.out_layers:
            output.append(layer(x))
        
        return output

    def initialize_tensorboard_manager(self, name):
        self.tensorBoardManager = TensorBoardManager(name)
        
    def set_device(self, device):
        self.device = device
    
    def set_dataset_manager(self, datasetManagaer):
        self.datasetManager = datasetManagaer

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_lambda(self, llambdas):
        self.llambdas = llambdas

    def get_multitask_loss(self, prediction_dict):
        loss = 0
        for k in prediction_dict.keys():
            loss += torch.sum(prediction_dict[k] * self.llambdas[k])
        return loss
    
    def set_hyperparameters(self, epochs, weighted = False):
        self.epochs = epochs
        self.weighted = weighted
        if self.weighted:
            self.datasetManager.compute_weights()

    def get_prediction_manager(self, loss_name):
        manager = Prediction(device = self.device)
        manager.select_loss(loss_name=loss_name)

        return manager

    def train_model(self):

        train_length = len(self.datasetManager.X_train)
        val_length = len(self.datasetManager.X_val)

        losses = Prediction(device = self.device).LOSSES

        distributional_prediction_train_manager = self.get_prediction_manager(loss_name=losses['cosine_dissimilarity'])
        distributional_prediction_val_manager = self.get_prediction_manager(loss_name=losses['cosine_dissimilarity'])

        hyperbolic_prediction_train_manager = self.get_prediction_manager(loss_name=losses['hyperbolic_distance'])
        hyperbolic_prediction_val_manager = self.get_prediction_manager(loss_name=losses['hyperbolic_distance'])
        
        if self.weighted:

            train_weights = self.datasetManager.get_weights(label = 'Train')
            val_weights = self.datasetManager.get_weights(label = 'Val')

            distributional_prediction_train_manager.set_weight(train_weights)
            hyperbolic_prediction_train_manager.set_weight(train_weights)
            
            distributional_prediction_val_manager.set_weight(val_weights)
            hyperbolic_prediction_val_manager.set_weight(val_weights)

            train_weights_sum = 0
            val_weights_sum = 0

        for epoch in range(self.epochs):
            train_it = iter(self.datasetManager.trainloader)
            val_it = iter(self.datasetManager.valloader)
            
            train_loss_SUM = 0
            val_loss_SUM = 0
            
            distributional_train_loss_SUM = 0
            distributional_val_loss_SUM = 0
            
            hyperbolic_train_loss_SUM = 0
            hyperbolic_val_loss_SUM = 0
            
            for batch_iteration in range(len(self.datasetManager.trainloader)):
                x, labels, targets = next(train_it)
                
                ######################
                ####### TRAIN ########
                ######################
                
                
                self.optimizer.zero_grad()
                
                self.train()
                
                output = self(x)

                distributional_prediction_train_manager.set_prediction(predictions = output[0],
                                                                 true_values = targets['distributional'], 
                                                                 labels = labels)

                hyperbolic_prediction_train_manager.set_prediction(predictions = output[1], 
                                                             true_values = targets['hyperbolic'],
                                                             labels = labels)                


                distributional_train_loss = distributional_prediction_train_manager.compute_loss()
                
                hyperbolic_train_loss = hyperbolic_prediction_train_manager.compute_loss()

                distributional_train_loss_SUM += torch.sum(distributional_train_loss * self.llambdas['distributional']).item()
                hyperbolic_train_loss_SUM += torch.sum(hyperbolic_train_loss * (self.llambdas['hyperbolic'])).item()
                
                train_loss = self.get_multitask_loss({'distributional': distributional_train_loss, 
                                                      'hyperbolic': hyperbolic_train_loss})
                
                train_loss_SUM += train_loss.item()
                if self.weighted:
                    batch_weights = distributional_prediction_train_manager.get_batch_weights()
                    train_weights_sum += torch.sum(batch_weights)

                train_loss.backward()
                self.optimizer.step()
           
            else:

                ######################
                ######## VAL #########
                ######################
                
                with torch.no_grad():
                    self.eval()   
                    
                    for batch_iteration in range(len(self.datasetManager.valloader)):
                        x, labels, targets = next(val_it)


                        output = self(x)

                        distributional_prediction_val_manager.set_prediction(predictions = output[0],
                                                                        true_values = targets['distributional'],
                                                                        labels = labels)

                        hyperbolic_prediction_val_manager.set_prediction(predictions = output[1], 
                                                                    true_values = targets['hyperbolic'],
                                                                    labels = labels) 
                        
                        distributional_val_loss = distributional_prediction_val_manager.compute_loss()
                        
                        hyperbolic_val_loss = hyperbolic_prediction_val_manager.compute_loss()
                        
                        distributional_val_loss_SUM += torch.sum(distributional_val_loss * self.llambdas['distributional']).item()
                        hyperbolic_val_loss_SUM += torch.sum(hyperbolic_val_loss * (self.llambdas['hyperbolic'])).item()
                        
                        val_loss = self.get_multitask_loss({'distributional': distributional_val_loss, 
                                                            'hyperbolic': hyperbolic_val_loss})
                        
                        val_loss_SUM += val_loss.item()

                        if self.weighted:
                            batch_weights = distributional_prediction_val_manager.get_batch_weights()
                            val_weights_sum += torch.sum(batch_weights)
            
            if not self.weighted:
                train_loss_value = train_loss_SUM/train_length
                val_loss_value = val_loss_SUM/val_length

                hyperbolic_train_loss_value = hyperbolic_train_loss_SUM/train_length
                hyperbolic_val_loss_value = hyperbolic_val_loss_SUM/val_length

                distributional_train_loss_value = distributional_train_loss_SUM/train_length
                distributional_val_loss_value = distributional_val_loss_SUM/val_length

            else:
                train_loss_value = train_loss_SUM/train_weights_sum
                val_loss_value = val_loss_SUM/val_weights_sum

                hyperbolic_train_loss_value = hyperbolic_train_loss_SUM/train_weights_sum
                hyperbolic_val_loss_value = hyperbolic_val_loss_SUM/val_weights_sum

                distributional_train_loss_value = distributional_train_loss_SUM/train_weights_sum
                distributional_val_loss_value = distributional_val_loss_SUM/val_weights_sum
                

            losses_dict = {'Losses/Hyperbolic Losses': {'Train': hyperbolic_train_loss_value, 
                                                 'Val': hyperbolic_val_loss_value},
                           'Losses/Distributional Losses':  {'Train': distributional_train_loss_value,
                                                      'Val': distributional_val_loss_value},
                           'Losses/MTL-Losses': {'Train': train_loss_value,
                                          'Val': val_loss_value}
            }


            self.log_losses(losses_dict = losses_dict, epoch = epoch + 1)

            print('{:^15}'.format('epoch {:^3}/{:^3}'.format(epoch, self.epochs)))
            print('{:^15}'.format('Train loss: {:.4f}, Val loss: {:.4f}'.format(train_loss_value, val_loss_value)
                                                                            
                                )
                )
    def log_losses(self, losses_dict, epoch):
        for k, subdict in losses_dict.items():
            list_of_losses = [subdict['Train'], subdict['Val']]

            self.tensorBoardManager.log_losses(main_label= k, 
                                                list_of_losses = list_of_losses,
                                                list_of_names = ['Train', 'Val'], 
                                                epoch = epoch)

    
    def type_prediction_on_test(self, topn = None):
        '''
        compute results on testset in type prediction task
        '''

        test_predictions = self(torch.tensor(self.datasetManager.X_test, device=self.device))
        labels = self.datasetManager.Y_test
        entities = self.datasetManager.E_test

        self.topn = topn

        for space, preds in zip(['distributional', 'hyperbolic'], test_predictions):

            print(' ...evaluating test predictions in {} space... '.format(space))

            if self.device == torch.device("cuda"): 
                preds = preds.detach().cpu().numpy()   
            else:
                preds = preds.numpy()

            self.emb = Embedding()            
            self.emb.set_embedding(self.datasetManager.concept_embeddings[space])
            self.emb.set_name(space)
            
            if space == 'distributional':
                self.emb.set_distance(cosine_dissimilarity)
                self.emb.set_centroid_method(vector_mean)
            elif space == 'hyperbolic':
                self.emb.set_distance(hyper_distance)
                self.emb.set_centroid_method(hyperbolic_midpoint)
            
            print('occurrence {}'.format(space))
            micro, macro, concept_accuracies = self.occurrence_level_prediction(predictions = preds, 
                                                                                labels = labels)
            self.save_results(micro = micro, 
                                macro = macro, 
                                concept_accuracies = concept_accuracies, 
                                level_name = 'Occurrence Level in {} space'.format(space))
            print('entity {}'.format(space))
            micro, macro, concept_accuracies = self.entity_level_prediction(predictions = preds, 
                                                                            labels = labels, 
                                                                            entities = entities)
            
            self.save_results(micro = micro, 
                                macro = macro, 
                                concept_accuracies = concept_accuracies, 
                                level_name = 'Entity Level in {} space'.format(space))
            print('concept1 {}'.format(space))
            micro, macro, concept_accuracies = self.concept_level_prediction(predictions = preds,
                                                                                labels = labels)

            self.save_results(micro = micro, 
                                macro = macro, 
                                concept_accuracies = concept_accuracies, 
                                level_name = 'Concept Level (induce from occurrencies) in {} space'.format(space))
            print('concept2 {}'.format(space))
            micro, macro, concept_accuracies = self.concept_level_prediction(predictions = preds,
                                                                                labels = labels,
                                                                                entities = entities)

            self.save_results(micro = micro, 
                                macro = macro, 
                                concept_accuracies = concept_accuracies, 
                                level_name = 'Concept Level (induce from entities) in {} space'.format(space))

    def compute_metrics(self, total, concept_accuracies, elements_number_for_concept, predictions, labels):
        corrects_n = 0
        
        for pred, label in zip(predictions, labels):
            neigh = self.emb.get_neigh(vector = pred, top = self.topn)
            if label in neigh:
                corrects_n += 1
                concept_accuracies[label] += 1
        
        concept_accuracies = {s: concept_accuracies[s]/elements_number_for_concept[s] for s in set(labels)}
        micro_average_accuracy = corrects_n/total
        macro_average_accuracy = np.mean(list(concept_accuracies.values()))

        return micro_average_accuracy, macro_average_accuracy, concept_accuracies

    def occurrence_level_prediction(self, predictions, labels):

        corrects_n = 0
        total_n = len(labels)
        concept_accuracies = {s:0 for s in set(labels)}
        concept_number = {s:len([l for l in labels if l == s]) for s in set(labels)}

        return self.compute_metrics(total=total_n,
                                    concept_accuracies=concept_accuracies,
                                    elements_number_for_concept=concept_number,
                                    predictions= predictions,
                                    labels = labels)        

    def entity_level_prediction(self, predictions, labels, entities):
        entities_dict = {e: [v for v, entity in zip(predictions, entities) if entity == e] for e in set(entities)}
        entities_labels = {e: l for e, l in zip(entities, labels)}

        entities_predictions = self.induce_vector(entities_dict)

        if entities_predictions.keys() != entities_labels.keys():
            raise Exception('keys does not match, check the induce method') from e
        
        corrects_n = 0
        total_n = len(set(entities))
        concept_accuracies = {s:0 for s in set(labels)}
        concept_number = {s:len([e for e in labels if e == s]) for s in set(labels)}

        entity_predictions_list = list(entities_predictions.values())
        entity_labels_list = list(entities_labels.values())  

        return self.compute_metrics(total=total_n,
                                    concept_accuracies = concept_accuracies,
                                    elements_number_for_concept=concept_number,
                                    predictions = entity_predictions_list,
                                    labels = entity_labels_list)


    def concept_level_prediction(self, predictions, labels, entities = None):
        if entities:
            # induce entities and then induce concepts from entities
            entities_dict = {e: [v for v, entity in zip(predictions, entities) if entity == e] for e in set(entities)}
            entities_labels = {e: l for e, l in zip(entities, labels)}

            entities_vectors = self.induce_vector(entities_dict)

            if entities_vectors.keys() != entities_labels.keys():
                raise Exception('keys does not match, check the induce method') from e
            
            predictions = list(entities_vectors.values())
            labels = list(entities_labels.values())  

        concepts_dict = {s: [v for v, label in zip(predictions, labels) if label == s] for s in set(labels)}

        concept_vectors = self.induce_vector(concepts_dict)
        corrects_n = 0
        total_n = len(set(labels))
        concept_accuracies = {s:0 for s in set(labels)}
        concept_number = {s: 1 for s in set(labels)}

        concept_vectors_list = list(concept_vectors.values())
        concept_labels_list = list(concept_vectors.keys())  


        return self.compute_metrics(total=total_n,
                                    concept_accuracies = concept_accuracies,
                                    elements_number_for_concept=concept_number,
                                    predictions = concept_vectors_list,
                                    labels = concept_labels_list)

    
    def induce_vector(self, clusters_dict):
        return {e: self.emb.get_centroid(v) for e, v in clusters_dict.items()}


    def save_results(self, micro, macro, concept_accuracies, level_name):
        with open('results/results.txt', 'a') as out:
            out.write('------------------------ topn: {}--------------------------\n'.format(self.topn))
            out.write('\n{} results: \n'.format(level_name))
            out.write('\t micro average accuracy: {:.4f}\n'.format(micro))
            out.write('\t macro average accuracy: {:.4f}\n'.format(macro))

            out.write('Accuracy for each concept:\n')

            for concept, acc in concept_accuracies.items():
                out.write('\t{:15}: {:4.2f}\n'.format(concept, acc))



class Embedding:

    def set_name(self, name):
        self.name = name

    def set_embedding(self, embedding):
        self.embedding = embedding

    def set_distance(self, distance):
        self.distance = distance

    def get_vector(self, label):
        try:
            return self.embedding[label]    
        except:
            raise Exception('label not this embedding ({}'.format(self.name)) from e

    def get_neigh(self, vector, top = None):
        if not top:
            top = len(self.embedding)
        neigh, similarities = self.find_neigh(vector, top)
        return neigh

    def find_neigh(self, vec, topn):
        dists = {}
        for k, v in self.embedding.items():
            dists[k] = self.distance(v, vec)
        return [a for a, b in sorted(dists.items(), key=lambda item: item[1])[:topn]], [b for a, b in sorted(dists.items(), key=lambda item: item[1])[:topn]]

    def set_centroid_method(self, method):
        self.centroid_method = method

    def get_centroid(self, vectors):
        return self.centroid_method(vectors)

class Prediction:

    def __init__(self, device = None, weighted = False):
        self.LOSSES = {'cosine_dissimilarity': 'COSD',
                       'hyperbolic_distance': 'HYPD',
                       'regularized_hyperbolic_distance': 'RHYPD',
                       'hyperboloid_distance' : 'LORENTZD'
        }
        self.device = device
        self.weighted = weighted

    def set_weight(self, weights):
        self.weights = weights

    def set_prediction(self, predictions, true_values, labels):
        self.predictions = predictions
        self.true_values = true_values
        self.labels = labels
    
    def select_loss(self, loss_name):
        if loss_name == self.LOSSES['cosine_dissimilarity']:
            self.selected_loss = cosineLoss(device = self.device)
        elif loss_name == self.LOSSES['hyperbolic_distance']:
            self.selected_loss = poincareDistanceLoss(device = self.device)
        elif loss_name == self.LOSSES['regularized_hyperbolic_distance']:
            self.selected_loss = regularizedPoincareDistanceLoss(device = self.device)
        # elif loss_name == self.LOSSES['hyperboloid_distance']:
        #     self.selected_loss = lorentzDistanceLoss(device = self.device)

    def compute_loss(self):
        loss = self.selected_loss.compute_loss(true = self.true_values,
                                                   pred = self.predictions)
        if not self.weighted:
            return loss
        else:
            batch_weights = self.get_batch_weights()
            return loss * batch_weights
    
    def get_batch_weights(self):
        batch_weights = [self.weights[l.item()] for l in self.labels]
        return torch.tensor(batch_weights, dtype=torch.float64, device = self.device)


class Loss(ABC):

    def __init__(self, device):
        self.device = device

    @abstractmethod
    def compute_loss(self):
        pass

class cosineLoss(Loss):
    def compute_loss(self, true, pred):
        cossim = torch.nn.CosineSimilarity(dim = 1)
        return 1 - cossim(true, pred)

class poincareDistanceLoss(Loss):
    def compute_loss(self, true, pred):
        numerator = 2 * torch.norm(true - pred, dim = 1)**2

        pred_norm = torch.norm(pred, dim = 1)**2
        true_norm = torch.norm(true, dim = 1)**2

        left_denom = 1 - pred_norm
        right_denom = 1 - true_norm
        
        denom = left_denom * right_denom

        frac = numerator/denom
        acos = self.acosh(1  + frac)
        
        return acos

    def acosh(self, x):
        return torch.log(x + torch.sqrt(x**2-1))

    def mse(self, y_pred, y_true):    
        mse_loss = nn.MSELoss()
        return mse_loss(y_pred, y_true)

class regularizedPoincareDistanceLoss(poincareDistanceLoss):

    def set_regularization(self, regul):
        self.regul = regul

    def compute_loss(self, true, pred):
        acos = super().compute_loss(true = true, pred = pred)

        l0 = torch.tensor(1., device = self.device)
        l1 = torch.tensor(1., device = self.device)
        
        if sum(self.regul) > 1:
            
            true_perm = true[torch.randperm(true.size()[0])]
            
            l0 = torch.abs(super().compute_loss(pred, true_perm) - super.compute_loss(true, true_perm))
            l1 = self.mse(pred, true)
        
        return acos**self.regul[2] + l0 * self.regul[0] + l1 * self.regul[1]