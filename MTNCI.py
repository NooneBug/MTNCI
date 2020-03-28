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
from collections import defaultdict
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.autograd import Variable
from tqdm import tqdm

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
    
    def set_dataset_manager(self, datasetManager):
        self.datasetManager = datasetManager

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
        manager = Prediction(device = self.device, weighted=self.weighted)
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


        for epoch in range(self.epochs):
            train_it = iter(self.datasetManager.trainloader)
            val_it = iter(self.datasetManager.valloader)
            
            train_loss_SUM = 0
            val_loss_SUM = 0
            
            distributional_train_loss_SUM = 0
            distributional_val_loss_SUM = 0
            
            hyperbolic_train_loss_SUM = 0
            hyperbolic_val_loss_SUM = 0
            
            train_weights_sum = 0
            val_weights_sum = 0
            
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


            self.checkpointManager(val_loss_value)

            self.log_losses(losses_dict = losses_dict, epoch = epoch + 1)

            print('{:^15}'.format('epoch {:^3}/{:^3}'.format(epoch, self.epochs)))
            print('{:^15}'.format('Train loss: {:.4f}, Val loss: {:.4f}'.format(train_loss_value, val_loss_value)
                                                                            
                                )
                )

    def checkpointManager(self, val_loss_value):
        try:
            a = self.min_loss
        except:
            self.min_loss = val_loss_value

        if val_loss_value <= self.min_loss:
            self.min_loss = val_loss_value
            torch.save({
                'model_state_dict' : self.state_dict()
            }, self.checkpoint_path)

    def set_checkpoint_path(self, checkpoint_path):
        self.checkpoint_path = checkpoint_path

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
        
        self.topn = topn
        checkpoint = torch.load(self.checkpoint_path)

        self.load_state_dict(checkpoint['model_state_dict'])
        test_predictions = self(torch.tensor(self.datasetManager.X_test, device=self.device))
        labels = self.datasetManager.Y_test
        entities = self.datasetManager.E_test

        return self.compute_prediction(test_predictions, labels, entities)


    def compute_prediction(self, test_predictions, labels, entities):

        for space, preds in zip(['distributional', 'hyperbolic'], test_predictions):
            
            self.eval()

            print(' ...evaluating test predictions in {} space... '.format(space))

            # if self.device == torch.device("cuda"): 
            #     preds = preds.detach().cpu().numpy()   
            # else:
            #     preds = preds.numpy()

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
            recalls, precisions, fmeasures = self.occurrence_level_prediction(predictions = preds, 
                                                                                labels = labels)
            self.save_results(recalls = recalls, 
                                precisions = precisions, 
                                fmeasures = fmeasures, 
                                level_name = 'Occurrence Level in {} space'.format(space))
            print('entity {}'.format(space))
            recalls, precisions, fmeasures = self.entity_level_prediction(predictions = preds, 
                                                                            labels = labels, 
                                                                            entities = entities)
            
            self.save_results(recalls = recalls, 
                                precisions = precisions, 
                                fmeasures = fmeasures, 
                                level_name = 'Entity Level in {} space'.format(space))
            print('concept1 {}'.format(space))
            recalls, precisions, fmeasures = self.concept_level_prediction(predictions = preds,
                                                                                labels = labels)

            self.save_results(recalls = recalls, 
                                precisions = precisions, 
                                fmeasures = fmeasures, 
                                level_name = 'Concept Level (induce from occurrencies) in {} space'.format(space))
            print('concept2 {}'.format(space))
            recalls, precisions, fmeasures = self.concept_level_prediction(predictions = preds,
                                                                                labels = labels,
                                                                                entities = entities)

            self.save_results(recalls = recalls, 
                                precisions = precisions, 
                                fmeasures = fmeasures,  
                                level_name = 'Concept Level (induce from entities) in {} space'.format(space))

    def compute_metrics(self, total, concept_accuracies, elements_number_for_concept, predictions, labels):
        precision_at_n = {t: 0 for t in self.topn}
        recall_at_n = {t: 0 for t in self.topn}
        f_measure = {t: 0 for t in self.topn}

        concept_predicted = defaultdict(int)
        correct_prediction = defaultdict(int)
        corrects_n = {t: 0 for t in self.topn}

        for pred, label in zip(predictions, labels):
            neigh = self.emb.get_neigh(vector = pred, top = max(self.topn))

            for t in self.topn:
                new_neigh = neigh[:t]
                if label in new_neigh:
                    corrects_n[t] += 1
                    concept_accuracies[t][label] += 1
            
                if t == 1:
                    concept_predicted[new_neigh[0]] += 1
                    if label in new_neigh:
                        correct_prediction[new_neigh[0]] += 1

        micro_average_recall, macro_average_recall = {}, {}

        for t in self.topn:
            recall_at_n[t] = {s: (concept_accuracies[t][s]/elements_number_for_concept[s], elements_number_for_concept[s]) for s in set(labels)}
            micro_average_recall[t] = corrects_n[t]/total
            macro_average_recall[t] = np.mean([p[0] for p in recall_at_n[t].values()])

        micro_average_precision, macro_average_precision = {}, {}

        precision_at_n[1] = {s: (correct_prediction[s]/concept_predicted[s] if concept_predicted[s] != 0 else 0, concept_predicted[s]) for s in set(labels)}
        micro_average_precision[1] = sum([c for c in correct_prediction.values()])/ sum([c for c in concept_predicted.values()])
        macro_average_precision[1] = np.mean([p[0] for p in precision_at_n[1].values()]) 

        micro_average_fmeasure, macro_average_fmeasure = defaultdict(int), defaultdict(int)
        
        f_measure[1] = {s: self.f1(precision_at_n[1][s][0], recall_at_n[1][s][0]) for s in set(labels)}
        macro_average_fmeasure[1] = self.f1(macro_average_precision[1], macro_average_recall[1])
        micro_average_fmeasure[1] = self.f1(micro_average_precision[1], micro_average_recall[1])

        recalls = self.make_measure_dict(micro_average_recall, macro_average_recall, recall_at_n)
        precisions = self.make_measure_dict(micro_average_precision, macro_average_precision, precision_at_n)
        fmeasures = self.make_measure_dict(micro_average_fmeasure, macro_average_fmeasure, f_measure)

        return recalls, precisions, fmeasures                


    def make_measure_dict(self, micro_average, macro_average, at):
        return {'micro': micro_average, 'macro': macro_average, 'at' : at}

    def f1(self, p, r):
        if r == 0.:
            return 0.
        return 2 * p * r / float(p + r)

    def occurrence_level_prediction(self, predictions, labels):

        corrects_n = 0
        total_n = len(labels)
        concept_accuracies = {t: {s:0 for s in set(labels)} for t in self.topn}
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
        concept_accuracies =  {t: {s:0 for s in set(labels)} for t in self.topn}
        concept_number = {s:len([e for e in labels if e == s]) for s in set(labels)}

        entity_predictions_list = list(entities_predictions.values())
        entity_labels_list = [entities_labels[e] for e in entities_predictions.keys()]   

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
            labels = [entities_labels[e] for e in entities_vectors.keys()] 

        concepts_dict = {s: [v for v, label in zip(predictions, labels) if label == s] for s in set(labels)}

        concept_vectors = self.induce_vector(concepts_dict)

        total_n = len(set(labels))
        concept_accuracies =  {t: {s:0 for s in set(labels)} for t in self.topn}
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


    def save_results(self, precisions, recalls, fmeasures, level_name):
        with open('results/results.txt', 'a') as out:
            for t in self.topn:
                out.write('------------------------ topn: {}--------------------------\n'.format(t))
                out.write('\n{} results: \n'.format(level_name))
                
                out.write('\t micro average recall: {:.4f}\n'.format(recalls['micro'][t]))
                out.write('\t macro average recall: {:.4f}\n'.format(recalls['macro'][t]))

                if t == 1:
                    out.write('\t micro average precision: {:.4f}\n'.format(precisions['micro'][t]))
                    out.write('\t macro average precision: {:.4f}\n'.format(precisions['macro'][t]))

                    out.write('\t micro average fmeasure: {:.4f}\n'.format(fmeasures['micro'][t]))
                    out.write('\t macro average fmeasure: {:.4f}\n'.format(fmeasures['macro'][t]))
                
                out.write('Metrics for each concept:\n')

                keys = sorted(list(recalls['at'][t].keys()))

                out.write('\t{:35}| {:10}| {:10}| {:10}| {:4}| {:4}\n'.format('Concept', 
                                                                    'recall', 
                                                                    'precision',
                                                                    'fmeasure',
                                                                    '#', 
                                                                    'predictions (for precision)'))
                
                for k in keys:
                    if t == 1:
                        out.write('\t{:35}: {:10.2f}, {:10.2f}, {:10.2f}, {:4}, {:4}\n'.format(k,
                                                                                recalls['at'][t][k][0],
                                                                                precisions['at'][t][k][0],
                                                                                fmeasures['at'][t][k], 
                                                                                recalls['at'][t][k][1],
                                                                                precisions['at'][t][k][1]
                                                                                ))
                    else:
                        out.write('\t{:35}: {:10.2f}, {:10.2f}, {:10.2f}, {:4}, {:4}\n'.format(k,
                                                                                recalls['at'][t][k][0],
                                                                                0,
                                                                                0, 
                                                                                recalls['at'][t][k][1],
                                                                                ' '
                                                                                ))


    def get_model(self):
        return self

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


class ShimaokaMTNCI(MTNCI):
    
    def __init__(self, argss, vocabs, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        CHAR_VOCAB = 'char'        
        self.word_lut = nn.Embedding(vocabs["token"].size_of_word2vecs(), 
                                     argss.emb_size,
                                     padding_idx=0).cuda()
        
        self.mention_encoder = MentionEncoder(vocabs[CHAR_VOCAB], argss).cuda()
        self.context_encoder = ContextEncoder(argss).cuda()
        self.feature_len = argss.context_rnn_size * 2 + argss.emb_size + argss.char_emb_size
    
    def init_params(self, word2vec):
        self.word_lut.weight.data.copy_(word2vec)
        self.word_lut.weight.requires_grad = False
        
    def forward(self, input):
        contexts, positions, context_len = input[0], input[1].double(), input[2]
        mentions, mention_chars = input[3], input[4]
        type_indexes = input[5]
                
        mention_vec = self.mention_encoder(mentions, mention_chars, self.word_lut)
        
        context_vec, attn = self.context_encoder(contexts, positions, context_len, self.word_lut)

        input_vec = torch.cat((mention_vec, context_vec), dim=1)
        
        return super().forward(input_vec)
    

    def train_model(self):
        train_length = len(self.datasetManager.Y_train)
        val_length = len(self.datasetManager.Y_val)

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


        for epoch in range(self.epochs):            
            train_loss_SUM = 0
            val_loss_SUM = 0
            
            distributional_train_loss_SUM = 0
            distributional_val_loss_SUM = 0
            
            hyperbolic_train_loss_SUM = 0
            hyperbolic_val_loss_SUM = 0
            
            train_weights_sum = 0
            val_weights_sum = 0

            bar = tqdm(total= len(self.datasetManager.train_batched_datas))
            bar.set_description("Training on train set")
            
            for batch_iteration in range(len(self.datasetManager.train_batched_datas)):
                x = self.datasetManager.train_batched_datas[batch_iteration]
                labels = self.datasetManager.batched_Y_train[batch_iteration]
                targets = self.datasetManager.batched_aligned_train[batch_iteration][0]
                
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

                bar.update(1)
           
            else:

                ######################
                ######## VAL #########
                ######################
                
                with torch.no_grad():
                    self.eval()   
                    
                    bar = tqdm(total= len(self.datasetManager.val_batched_datas))
                    bar.set_description("Evaluating validation set")
                    for batch_iteration in range(len(self.datasetManager.val_batched_datas)):
                        x = self.datasetManager.val_batched_datas[batch_iteration]
                        labels = self.datasetManager.batched_Y_val[batch_iteration]
                        targets = self.datasetManager.batched_aligned_val[batch_iteration][0]


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
                        bar.update(1)
            
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


            self.checkpointManager(val_loss_value)

            self.log_losses(losses_dict = losses_dict, epoch = epoch + 1)

            print('{:^15}'.format('epoch {:^3}/{:^3}'.format(epoch, self.epochs)))
            print('{:^15}'.format('Train loss: {:.4f}, Val loss: {:.4f}'.format(train_loss_value, val_loss_value)
                                                                            
                                )
                )

    def type_prediction_on_test(self, topn, test_data, labels, entities):

        self.topn = topn

        x = [[], []]
        i = 0

        for i in range(len(test_data['data'])):
            t = test_data['data'][i]
            pred = self(t)
            x[0].extend(pred[0].detach().cpu().numpy())
            x[1].extend(pred[1].detach().cpu().numpy())
        
        self.compute_prediction(x, labels, entities)



class CharEncoder(nn.Module):
    def __init__(self, char_vocab, args):
        super(CharEncoder, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        conv_dim_input = 100
        filters = 5
        self.char_W = nn.Embedding(char_vocab.size(), conv_dim_input, padding_idx=0)
        self.conv1d = nn.Conv1d(conv_dim_input, args.char_emb_size, filters)  # input, output, filter_number

    def forward(self, span_chars):
        char_embed = self.char_W(span_chars).transpose(1, 2)  # [batch_size, char_embedding, max_char_seq]
        conv_output = [self.conv1d(char_embed)]  # list of [batch_size, filter_dim, max_char_seq, filter_number]
        conv_output = [F.relu(c) for c in conv_output]  # batch_size, filter_dim, max_char_seq, filter_num
        cnn_rep = [F.max_pool1d(i, i.size(2)) for i in conv_output]  # batch_size, filter_dim, 1, filter_num
        cnn_output = torch.squeeze(torch.cat(cnn_rep, 1), 2)  # batch_size, filter_num * filter_dim, 1
        return cnn_output

class MentionEncoder(nn.Module):

    def __init__(self, char_vocab, args):
        super(MentionEncoder, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.char_encoder = CharEncoder(char_vocab, args)
        self.attentive_weighted_average = SelfAttentiveSum(args.emb_size, 1)
        self.dropout = nn.Dropout(args.mention_dropout)

    def forward(self, mentions, mention_chars, word_lut):
        mention_embeds = word_lut(mentions)             # batch x mention_length x emb_size

        weighted_avg_mentions, _ = self.attentive_weighted_average(mention_embeds)
        char_embed = self.char_encoder(mention_chars)
        output = torch.cat((weighted_avg_mentions, char_embed), 1)
        return self.dropout(output).cuda()


class ContextEncoder(nn.Module):

    def __init__(self, args):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.emb_size = args.emb_size
        self.pos_emb_size = args.positional_emb_size
        self.rnn_size = args.context_rnn_size
        self.hidden_attention_size = 100
        super(ContextEncoder, self).__init__()
        self.pos_linear = nn.Linear(1, self.pos_emb_size)
        self.context_dropout = nn.Dropout(args.context_dropout)
        self.rnn = nn.LSTM(self.emb_size + self.pos_emb_size, self.rnn_size, bidirectional=True, batch_first=True)
        self.attention = SelfAttentiveSum(self.rnn_size * 2, self.hidden_attention_size) # x2 because of bidirectional

    def forward(self, contexts, positions, context_len, word_lut, hidden=None):
        """
        :param contexts: batch x max_seq_len
        :param positions: batch x max_seq_len
        :param context_len: batch x 1
        """
        positional_embeds = self.get_positional_embeddings(positions)   # batch x max_seq_len x pos_emb_size
        ctx_word_embeds = word_lut(contexts)                            # batch x max_seq_len x emb_size
        ctx_embeds = torch.cat((ctx_word_embeds, positional_embeds), 2)

        ctx_embeds = self.context_dropout(ctx_embeds)

        rnn_output = self.sorted_rnn(ctx_embeds, context_len)

        return self.attention(rnn_output)

    def get_positional_embeddings(self, positions):
        """ :param positions: batch x max_seq_len"""
        pos_embeds = self.pos_linear(positions.view(-1, 1))                     # batch * max_seq_len x pos_emb_size
        return pos_embeds.view(positions.size(0), positions.size(1), -1)        # batch x max_seq_len x pos_emb_size

    def sorted_rnn(self, ctx_embeds, context_len):
        sorted_inputs, sorted_sequence_lengths, restoration_indices = self.sort_batch_by_length(ctx_embeds, context_len)
        packed_sequence_input = pack(sorted_inputs, sorted_sequence_lengths, batch_first=True)
        packed_sequence_output, _ = self.rnn(packed_sequence_input, None)
        unpacked_sequence_tensor, _ = unpack(packed_sequence_output, batch_first=True)
        return unpacked_sequence_tensor.index_select(0, restoration_indices)

    def sort_batch_by_length(self, tensor: torch.autograd.Variable, sequence_lengths: torch.autograd.Variable):
        """
        @ from allennlp
        Sort a batch first tensor by some specified lengths.

        Parameters
        ----------
        tensor : Variable(torch.FloatTensor), required.
            A batch first Pytorch tensor.
        sequence_lengths : Variable(torch.LongTensor), required.
            A tensor representing the lengths of some dimension of the tensor which
            we want to sort by.

        Returns
        -------
        sorted_tensor : Variable(torch.FloatTensor)
            The original tensor sorted along the batch dimension with respect to sequence_lengths.
        sorted_sequence_lengths : Variable(torch.LongTensor)
            The original sequence_lengths sorted by decreasing size.
        restoration_indices : Variable(torch.LongTensor)
            Indices into the sorted_tensor such that
            ``sorted_tensor.index_select(0, restoration_indices) == original_tensor``
        """

        if not isinstance(tensor, Variable) or not isinstance(sequence_lengths, Variable):
            raise ValueError("Both the tensor and sequence lengths must be torch.autograd.Variables.")

        sorted_sequence_lengths, permutation_index = sequence_lengths.sort(0, descending=True)
        sorted_tensor = tensor.index_select(0, permutation_index)
        # This is ugly, but required - we are creating a new variable at runtime, so we
        # must ensure it has the correct CUDA vs non-CUDA type. We do this by cloning and
        # refilling one of the inputs to the function.
        index_range = sequence_lengths.data.clone().copy_(torch.arange(0, len(sequence_lengths)))
        # This is the equivalent of zipping with index, sorting by the original
        # sequence lengths and returning the now sorted indices.
        index_range = Variable(index_range.long())
        _, reverse_mapping = permutation_index.sort(0, descending=False)
        restoration_indices = index_range.index_select(0, reverse_mapping)
        return sorted_tensor, sorted_sequence_lengths, restoration_indices


class SelfAttentiveSum(nn.Module):
    """
    Attention mechanism to get a weighted sum of RNN output sequence to a single RNN output dimension.
    """
    def __init__(self, embed_dim, hidden_dim):
        """
        :param embed_dim: in forward(input_embed), the size will be batch x seq_len x emb_dim
        :param hidden_dim:
        """
        super(SelfAttentiveSum, self).__init__()
        self.key_maker = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.key_rel = nn.ReLU()
        self.hidden_dim = hidden_dim
        self.key_output = nn.Linear(hidden_dim, 1, bias=False)
        self.key_softmax = nn.Softmax(dim=1)

    def forward(self, input_embed):     # batch x seq_len x emb_dim
        input_embed_squeezed = input_embed.view(-1, input_embed.size()[2])  # batch * seq_len x emb_dim
        k_d = self.key_maker(input_embed_squeezed)      # batch * seq_len x hidden_dim
        k_d = self.key_rel(k_d)
        if self.hidden_dim == 1:
            k = k_d.view(input_embed.size()[0], -1)     # batch x seq_len
        else:
            k = self.key_output(k_d).view(input_embed.size()[0], -1)  # (batch_size, seq_length)
        weighted_keys = self.key_softmax(k).view(input_embed.size()[0], -1, 1)  # batch x seq_len x 1
        weighted_values = torch.sum(weighted_keys * input_embed, 1)  # batch_size, embed_dim
        return weighted_values, weighted_keys