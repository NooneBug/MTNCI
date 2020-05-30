from torch import nn
import torch
import geoopt
from MTNCI.geooptModules import MobiusLinear, mobius_linear, create_ball
from torch.utils.data import Dataset, DataLoader
from geoopt.optim import RiemannianAdam
from MTNCI.tensorBoardManager import TensorBoardManager
from abc import ABC, abstractmethod
from preprocessing.utils import hyper_distance, hyperbolic_midpoint, cosine_dissimilarity, vector_mean
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import random
from MTNCI.ShimaokaModels import CharEncoder, MentionEncoder, SelfAttentiveSum, ContextEncoder
from preprocessing.utils import LOSSES
import copy
from torch.nn import Sigmoid

import sys 
sys.path.append('../figet-hyperbolic-space')
from figet.Constants import *

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

        for layer in self.fully:
            nn.init.xavier_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        
    def forward(self, x):
        for i in range(len(self.fully)):
            x = x.double()
            x = self.dropout(self.bns[i](self.leaky_relu(self.fully[i](x))))
        return x

class RegressionOutput(nn.Module):
    def __init__(self, hidden_dim, dims, manifold):
        super().__init__()
        self.out = nn.ModuleList()
        
        self.dropout = nn.Dropout(p=0.2).cuda()
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
            # x = self.dropout(self.leaky_relu(self.out[i](x)))
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

    def get_multitask_loss_vector(self, prediction_dict):
        loss = 0
        for k in prediction_dict.keys():
            loss += prediction_dict[k] * self.llambdas[k]
        return loss
    
    def set_hyperparameters(self, epochs, times = 1, perc = 1, patience = 50, weighted = False, regularized = False, beta = 500):
        self.epochs = epochs
        self.weighted = weighted
        self.regularized = regularized
        self.patience = patience
        self.early_stopping = False

        self.PERC = perc
        self.TIMES = times

        self.beta = beta

        if self.weighted:
            self.datasetManager.compute_weights()

    def set_regularization_params(self, regul_dict):
        self.regul_dict = regul_dict

    def get_prediction_manager(self, loss_name):
        manager = Prediction(device = self.device, weighted=self.weighted)
        manager.select_loss(loss_name=loss_name)

        return manager

    def set_losses(self, losses_name_dict):
        self.losses_name_dict = losses_name_dict

    def set_metrics(self, metric_name_dict):
        self.metric_name_dict = metric_name_dict

    def train_model(self):

        train_length = len(self.datasetManager.Y_train)
        val_length = len(self.datasetManager.Y_val)

        losses = Prediction(device = self.device).LOSSES

        # distributional_prediction_train_manager = self.get_prediction_manager(loss_name=losses['cosine_dissimilarity'])
        distributional_prediction_train_manager = self.get_prediction_manager(loss_name=self.losses_name_dict['distributional'])
        # distributional_prediction_val_manager = self.get_prediction_manager(loss_name=losses['cosine_dissimilarity'])
        distributional_prediction_val_manager = self.get_prediction_manager(loss_name=self.losses_name_dict['distributional'])

        # hyperbolic_prediction_train_manager = self.get_prediction_manager(loss_name=losses['regularized_hyperbolic_distance'])
        hyperbolic_prediction_train_manager = self.get_prediction_manager(loss_name=self.losses_name_dict['hyperbolic-train'])
        
        if self.regularized:
            print('in regularized')
            print('prediction_manager_getted')
            hyperbolic_prediction_train_manager.set_regularization_params(self.regul_dict)
            print('regul_setted')
        
        # hyperbolic_prediction_val_manager = self.get_prediction_manager(loss_name=losses['hyperbolic_distance'])
        # hyperbolic_prediction_val_manager = self.get_prediction_manager(loss_name=losses['normalized_hyperbolic_distance'])
        hyperbolic_prediction_val_manager = self.get_prediction_manager(loss_name=self.losses_name_dict['hyperbolic-val'])

        # hyperbolic_train_metric_manager = self.get_prediction_manager(loss_name=losses['hyperbolic_distance'])
        hyperbolic_train_metric_manager = self.get_prediction_manager(loss_name=self.metric_name_dict['hyperbolic'])
        # distributional_train_metric_manager = self.get_prediction_manager(loss_name=losses['cosine_dissimilarity'])
        distributional_train_metric_manager = self.get_prediction_manager(loss_name=self.metric_name_dict['distributional'])
        
        # hyperbolic_val_metric_manager = self.get_prediction_manager(loss_name=losses['hyperbolic_distance'])
        hyperbolic_val_metric_manager = self.get_prediction_manager(loss_name=self.metric_name_dict['hyperbolic'])
        # distributional_val_metric_manager = self.get_prediction_manager(loss_name=losses['cosine_dissimilarity'])
        distributional_val_metric_manager = self.get_prediction_manager(loss_name=self.metric_name_dict['distributional'])

        if self.weighted:

            train_weights = self.datasetManager.get_weights(label = 'Train')
            val_weights = self.datasetManager.get_weights(label = 'Val')

            distributional_prediction_train_manager.set_weight(train_weights)
            hyperbolic_prediction_train_manager.set_weight(train_weights)

            distributional_prediction_val_manager.set_weight(val_weights)
            hyperbolic_prediction_val_manager.set_weight(val_weights)

            hyperbolic_train_metric_manager.set_weight(train_weights)
            distributional_train_metric_manager.set_weight(train_weights)

            hyperbolic_val_metric_manager.set_weight(val_weights)
            distributional_val_metric_manager.set_weight(val_weights)

        for epoch in range(self.epochs):
            
            self.datasetManager.get_epoch_data(initialize=True)

            train_loss_SUM = 0
            val_loss_SUM = 0
            
            distributional_train_loss_SUM = 0
            distributional_val_loss_SUM = 0
            
            hyperbolic_train_loss_SUM = 0
            hyperbolic_val_loss_SUM = 0

            distributional_train_metric_SUM = 0
            hyperbolic_train_metric_SUM = 0
            
            distributional_val_metric_SUM = 0
            hyperbolic_val_metric_SUM = 0
            
            train_weights_sum = 0
            val_weights_sum = 0

            bar = tqdm(total=self.TIMES * self.datasetManager.get_data_batch_length(dataset='train'))
            for _ in range(self.TIMES):
                for batch_iteration in range(self.datasetManager.get_data_batch_length(dataset = 'train')):

                    if random.random() < self.PERC:
                        
                        x, labels, targets = self.datasetManager.get_epoch_data(dataset='train', batch_iteration = batch_iteration)

                        ######################
                        ####### TRAIN ########
                        ######################
                        # print('xs: {}'.format(x[0].shape))
                        # print('targets: {}'.format(targets['hyperbolic'].shape))
                        self.optimizer.zero_grad()
                        
                        self.train()
                        
                        output = self(x)

                        distributional_prediction_train_manager.set_prediction(predictions = output[0],
                                                                        true_values = targets['distributional'], 
                                                                        labels = labels)

                        hyperbolic_prediction_train_manager.set_prediction(predictions = output[1], 
                                                                    true_values = targets['hyperbolic'],
                                                                    labels = labels) 

                        distributional_train_metric_manager.set_prediction(predictions = output[0],
                                                                                        true_values = targets['distributional'], 
                                                                                        labels = labels)

                        hyperbolic_train_metric_manager.set_prediction(predictions = output[1], 
                                                                    true_values = targets['hyperbolic'],
                                                                    labels = labels)                

                        if self.weighted:
                            distributional_prediction_train_manager.compute_batch_weights()
                            batch_weights = distributional_prediction_train_manager.get_batch_weights()
                            train_weights_sum += torch.sum(batch_weights)

                            distributional_prediction_train_manager.set_batch_weights(batch_weights)
                            hyperbolic_prediction_train_manager.set_batch_weights(batch_weights)
                            distributional_train_metric_manager.set_batch_weights(batch_weights)
                            hyperbolic_train_metric_manager.set_batch_weights(batch_weights)

                        distributional_train_loss = distributional_prediction_train_manager.compute_loss()
                        
                        hyperbolic_train_loss = hyperbolic_prediction_train_manager.compute_loss()
                        
                        distributional_train_loss_SUM += torch.sum(distributional_train_loss * self.llambdas['distributional']).item()
                        hyperbolic_train_loss_SUM += torch.sum(hyperbolic_train_loss * self.llambdas['hyperbolic']).item()
                        

                        distributional_train_metric = distributional_train_metric_manager.compute_loss()
                        hyperbolic_train_metric = hyperbolic_train_metric_manager.compute_loss()

                        distributional_train_metric_SUM += torch.sum(distributional_train_metric).item()
                        hyperbolic_train_metric_SUM += torch.sum(hyperbolic_train_metric).item()

                        train_loss = self.get_multitask_loss({'distributional': distributional_train_loss, 
                                                            'hyperbolic': hyperbolic_train_loss})
                        
                        train_loss_SUM += train_loss.item()

                        train_loss.backward()
                        self.optimizer.step()
                    bar.update(1)
            bar.close()

            with torch.no_grad():
                self.eval()   
                bar_val = tqdm(total=self.datasetManager.get_data_batch_length(dataset='val'))
                # for batch_iteration in range(len(self.datasetManager.valloader)):
                for batch_iteration in range(self.datasetManager.get_data_batch_length(dataset='val')):
                    x, labels, targets = self.datasetManager.get_epoch_data(dataset='val', batch_iteration = batch_iteration)

                    ######################
                    ######## VAL #########
                    ######################

                    output = self(x)

                    distributional_prediction_val_manager.set_prediction(predictions = output[0],
                                                                    true_values = targets['distributional'],
                                                                    labels = labels)

                    hyperbolic_prediction_val_manager.set_prediction(predictions = output[1], 
                                                                true_values = targets['hyperbolic'],
                                                                labels = labels) 
                    
                    distributional_val_metric_manager.set_prediction(predictions = output[0],
                                                                            true_values = targets['distributional'], 
                                                                            labels = labels)

                    hyperbolic_val_metric_manager.set_prediction(predictions = output[1], 
                                                                        true_values = targets['hyperbolic'],
                                                                        labels = labels)                


                    if self.weighted:
                        distributional_prediction_val_manager.compute_batch_weights()
                        batch_weights = distributional_prediction_val_manager.get_batch_weights()
                        val_weights_sum += torch.sum(batch_weights)

                        distributional_prediction_val_manager.set_batch_weights(batch_weights)
                        hyperbolic_prediction_val_manager.set_batch_weights(batch_weights)
                        distributional_val_metric_manager.set_batch_weights(batch_weights)
                        hyperbolic_val_metric_manager.set_batch_weights(batch_weights)


                    distributional_val_loss = distributional_prediction_val_manager.compute_loss()
                    hyperbolic_val_loss = hyperbolic_prediction_val_manager.compute_loss()

                    distributional_val_loss_SUM += torch.sum(distributional_val_loss * self.llambdas['distributional']).item()
                    hyperbolic_val_loss_SUM += torch.sum(hyperbolic_val_loss * (self.llambdas['hyperbolic'])).item()
                    
                    distributional_val_metric = distributional_val_metric_manager.compute_loss()
                    hyperbolic_val_metric = hyperbolic_val_metric_manager.compute_loss()

                    distributional_val_metric_SUM += torch.sum(distributional_val_metric).item()
                    hyperbolic_val_metric_SUM += torch.sum(hyperbolic_val_metric).item()

                    val_loss = self.get_multitask_loss({'distributional': distributional_val_loss, 
                                                        'hyperbolic': hyperbolic_val_loss})
                    
                    val_loss_SUM += val_loss.item()

                    bar_val.update(1)

            bar_val.close()
            if not self.weighted:
                train_loss_value = train_loss_SUM/train_length
                val_loss_value = val_loss_SUM/val_length

                hyperbolic_train_loss_value = hyperbolic_train_loss_SUM/train_length
                hyperbolic_val_loss_value = hyperbolic_val_loss_SUM/val_length

                distributional_train_loss_value = distributional_train_loss_SUM/train_length
                distributional_val_loss_value = distributional_val_loss_SUM/val_length

                distributional_train_metric_value = distributional_train_metric_SUM/train_length
                hyperbolic_train_metric_value = hyperbolic_train_metric_SUM/train_length

                distributional_val_metric_value = distributional_val_metric_SUM/val_length
                hyperbolic_val_metric_value = hyperbolic_val_metric_SUM/val_length

            else:
                train_loss_value = train_loss_SUM/train_weights_sum
                val_loss_value = val_loss_SUM/val_weights_sum

                hyperbolic_train_loss_value = hyperbolic_train_loss_SUM/train_weights_sum
                hyperbolic_val_loss_value = hyperbolic_val_loss_SUM/val_weights_sum

                distributional_train_loss_value = distributional_train_loss_SUM/train_weights_sum
                distributional_val_loss_value = distributional_val_loss_SUM/val_weights_sum

                distributional_train_metric_value = distributional_train_metric_SUM/train_weights_sum
                hyperbolic_train_metric_value = hyperbolic_train_metric_SUM/train_weights_sum

                distributional_val_metric_value = distributional_val_metric_SUM/val_weights_sum
                hyperbolic_val_metric_value = hyperbolic_val_metric_SUM/val_weights_sum
                

            self.checkpointManager(val_loss_value, epoch)

            if self.early_stopping:
                break

            losses_dict = {'Losses/Hyperbolic Losses': {'Train': hyperbolic_train_loss_value, 
                                                 'Val': hyperbolic_val_loss_value},
                           'Losses/Distributional Losses':  {'Train': distributional_train_loss_value,
                                                      'Val': distributional_val_loss_value},
                           'Losses/MTL-Losses': {'Train': train_loss_value,
                                          'Val': val_loss_value}
            }


            metric_dict = {'Metrics/Hyperbolic Metrics': {'Train': hyperbolic_train_metric_value, 
                                                 'Val': hyperbolic_val_metric_value},
                           'Metrics/Distributional Metrics':  {'Train': distributional_train_metric_value,
                                                                'Val': distributional_val_metric_value}
                            }
            self.log_losses(losses_dict = losses_dict, epoch = epoch + 1)

            self.log_losses(losses_dict=metric_dict, epoch = epoch + 1)

            print('{:^25}'.format('epoch {:^3}/{:^3}'.format(epoch, self.epochs)))
            print('{:^25}'.format('Train loss: {:.4f}, Val loss: {:.4f}, Min loss: {:.4f} at epoch: {}'.format(train_loss_value, 
                                                                                                    val_loss_value, 
                                                                                                    self.min_loss, 
                                                                                                    self.best_epoch)))
            print('{:^25}'.format('T_MHD: {:.4f}, V_MHD:{:.4f}'.format(hyperbolic_train_metric_value, hyperbolic_val_metric_value)))
            print('{:^25}'.format('T_MDD: {:.4f}, V_MDD:{:.4f}'.format(distributional_train_metric_value, distributional_val_metric_value)))

    def checkpointManager(self, val_loss_value, epoch):
        try:
            a = self.min_loss
        except:
            self.min_loss = val_loss_value

        if val_loss_value <= self.min_loss:
            self.best_epoch = epoch
            self.min_loss = val_loss_value
            torch.save({
                'model_state_dict' : self.state_dict(),
                'epoch' : self.best_epoch 
            }, self.checkpoint_path)
        elif epoch == self.best_epoch + self.patience:
            print('early stopping')
            self.early_stopping = True


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

        try:
            print('loading model checkpoint at epoch {}'.format(checkpoint['epoch']))
        except:
            pass

        self.load_state_dict(checkpoint['model_state_dict'])
        
        self.eval()
        
        test_predictions = self(torch.tensor(self.datasetManager.X_test, device=self.device))
        labels = self.datasetManager.Y_test
        entities = self.datasetManager.E_test

        if self.device == torch.device("cuda"): 
            test_predictions[0] = test_predictions[0].detach().cpu().numpy()   
            test_predictions[1] = test_predictions[1].detach().cpu().numpy()   
        else:
            test_predictions[0] = test_predictions[0].numpy()
            test_predictions[1] = test_predictions[1].numpy()

        return self.compute_prediction(test_predictions, labels, entities)

    def compute_prediction(self, test_predictions, labels, entities):
        
        self.tsv = {'distributional':{'recalls': [],
                                'precisions': [],
                                'fmeasures': []},
                    'hyperbolic': {'recalls': [],
                                'precisions': [],
                                'fmeasures': []}
                    }

        for space, preds in zip(['distributional', 'hyperbolic'], test_predictions):
            print(' ...evaluating test predictions in {} space... '.format(space))
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
            
            self.fill_TSV(space = space, precisions=precisions, recalls=recalls, fmeasures=fmeasures)
            self.save_results(recalls = recalls, 
                                precisions = precisions, 
                                fmeasures = fmeasures, 
                                level_name = 'Occurrence Level in {} space'.format(space))

            print('entity {}'.format(space))
            recalls, precisions, fmeasures = self.entity_level_prediction(predictions = preds, 
                                                                            labels = labels, 
                                                                            entities = entities)
            
            self.fill_TSV(space = space, precisions=precisions, recalls=recalls, fmeasures=fmeasures)
            self.save_results(recalls = recalls, 
                                precisions = precisions, 
                                fmeasures = fmeasures, 
                                level_name = 'Entity Level in {} space'.format(space))
            
            print('concept1 {}'.format(space))
            recalls, precisions, fmeasures = self.concept_level_prediction(predictions = preds,
                                                                                labels = labels)
            
            self.fill_TSV(space = space, precisions=precisions, recalls=recalls, fmeasures=fmeasures)
            self.save_results(recalls = recalls, 
                                precisions = precisions, 
                                fmeasures = fmeasures, 
                                level_name = 'Concept Level (induce from occurrencies) in {} space'.format(space))
            
            print('concept2 {}'.format(space))
            recalls, precisions, fmeasures = self.concept_level_prediction(predictions = preds,
                                                                                labels = labels,
                                                                                entities = entities)
            self.fill_TSV(space = space, precisions=precisions, recalls=recalls, fmeasures=fmeasures)
            self.save_results(recalls = recalls, 
                                precisions = precisions, 
                                fmeasures = fmeasures,  
                                level_name = 'Concept Level (induce from entities) in {} space'.format(space))
            self.save_TSV(space = space)

    def compute_metrics(self, total, concept_accuracies, elements_number_for_concept, predictions, labels):
        precision_at_n = {t: 0 for t in self.topn}
        recall_at_n = {t: 0 for t in self.topn}
        f_measure = {t: 0 for t in self.topn}

        concept_predicted = defaultdict(int)
        correct_prediction = defaultdict(int)
        corrects_n = {t: 0 for t in self.topn}

        bar = tqdm(total=len(predictions))
        i = 0
        for pred, label in zip(predictions, labels):
            i += 1
            bar.set_description('getting results for prediction {}'.format(i))
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
            bar.update(1)
        bar.close()
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


        entities_and_labels = ['{}_{}'.format(e, l) for e, l in zip(entities, labels)]

        entities_dict = {e: [v for v, entity in zip(predictions, entities_and_labels) if entity == e] for e in set(entities_and_labels)}

        # entities_dict = {e: [v for v, entity in zip(predictions, entities) if entity == e] for e in set(entities)}
        entities_labels = {e: l for e, l in zip(entities_and_labels, labels)}

        entities_predictions = self.induce_vector(entities_dict)

        if entities_predictions.keys() != entities_labels.keys():
            raise Exception('keys does not match, check the induce method') from e
        
        corrects_n = 0
        total_n = len(set(entities))
        concept_accuracies =  {t: {s:0 for s in set(labels)} for t in self.topn}
        concept_number = {s:len([e for e, l in entities_labels.items() if l == s]) for s in set(labels)}

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
            entities_and_labels = ['{}_{}'.format(e, l) for e, l in zip(entities, labels)]
            entities_dict = {e: [v for v, entity in zip(predictions, entities_and_labels) if entity == e] for e in set(entities_and_labels)}

            # entities_dict = {e: [v for v, entity in zip(predictions, entities) if entity == e] for e in set(entities)}
            entities_labels = {e: l for e, l in zip(entities_and_labels, labels)}

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


    def set_results_paths(self, results_path, TSV_path):
        self.results_path = results_path
        self.TSV_path = TSV_path

    def fill_TSV(self, space, precisions, recalls, fmeasures):
        self.tsv[space]['recalls'].append(recalls['micro'][1])
        self.tsv[space]['recalls'].append(recalls['macro'][1])
        self.tsv[space]['precisions'].append(precisions['micro'][1])
        self.tsv[space]['precisions'].append(precisions['macro'][1])
        self.tsv[space]['fmeasures'].append(fmeasures['micro'][1])
        self.tsv[space]['fmeasures'].append(fmeasures['macro'][1])

    def save_TSV(self, space):
        if space == 'distributional':
            out = open(self.TSV_path, 'a')
        else:
            out = open(self.TSV_path, 'a')
        
        output = ''
        if space == 'hyperbolic':
            output += 'Hyperbolic\t\t\t\t\t\t\t\n'
            output += 'Micro\tMacro\tMicro\tMacro\tMicro\tMacro\tMicro\tMacro\n'
        for measure in ['precisions', 'recalls', 'fmeasures']:
            output += '\t'.join(['{:.4f}'.format(m) for m in self.tsv[space][measure]]) + '\n'
        out.write(output)        


    def save_results(self, precisions, recalls, fmeasures, level_name):
        with open(self.results_path, 'a') as out:
            for t in self.topn:
                out.write('------------------------ topn: {}--------------------------\n'.format(t))
                out.write('\n{} results: \n'.format(level_name))
                if t == 1:
                    out.write('\t micro average precision: {:.4f}\n'.format(precisions['micro'][t]))
                
                out.write('\t micro average recall: {:.4f}\n'.format(recalls['micro'][t]))

                if t == 1:
                    out.write('\t micro average fmeasure: {:.4f}\n'.format(fmeasures['micro'][t]))
                    out.write('\t macro average precision: {:.4f}\n'.format(precisions['macro'][t]))

                out.write('\t macro average recall: {:.4f}\n'.format(recalls['macro'][t]))
                if t == 1:
                    out.write('\t macro average fmeasure: {:.4f}\n'.format(fmeasures['macro'][t]))
                
                out.write('Metrics for each concept:\n')

                keys = sorted(list(recalls['at'][t].keys()))

                out.write('\t{:35}| {:10}| {:10}| {:10}| {:4}| {:4}\n'.format('Concept', 
                                                                    'recall', 
                                                                    'precision',
                                                                    'fmeasure',
                                                                    '#', 
                                                                    'predictions (for precision)'))
                
                bar = tqdm(total=len(keys))
                bar.set_description('Writing results for {}, top {}'.format(level_name, t))
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
                    bar.update()
                bar.close()

    def set_regularization_params(self, regul_dict):
        self.regul_dict = regul_dict


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

    def get_neigh(self, vector, top = None, include = 'all'):
        if not top:
            top = len(self.embedding)
        neigh, similarities = self.find_neigh(vector, top, include)
        return neigh

    def find_neigh(self, vec, topn, include = 'all'):
        dists = {}

        if include == 'coarse':
            include_list = COARSE
        elif include == 'fine':
            include_list = FINE
        elif include == 'ultrafine':
            include_list = set(self.embedding.keys()).difference(set(COARSE))
            include_list = include_list.difference(set(FINE))

        for k, v in self.embedding.items():
            if include == 'all':
                # print('v: {}'.format(v))
                # print('vec: {}'.format(vec))
                dists[k] = self.distance(v, vec)
            elif k in include_list:
                # print('v: {}'.format(v))
                # print('vec: {}'.format(vec))
                dists[k] = self.distance(v, vec)

            
        return [a for a, b in sorted(dists.items(), key=lambda item: item[1])[:topn]], [b for a, b in sorted(dists.items(), key=lambda item: item[1])[:topn]]

    def set_centroid_method(self, method):
        self.centroid_method = method

    def get_centroid(self, vectors):
        return self.centroid_method(vectors)

    
class Prediction:

    def __init__(self, device = None, weighted = False):
        self.LOSSES = LOSSES
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
        elif loss_name == self.LOSSES['normalized_hyperbolic_distance']:
            self.selected_loss = normalizedPoincareDistanceLoss(device = self.device)
        elif loss_name == self.LOSSES['regularized_hyperbolic_distance']:
            self.selected_loss = regularizedPoincareDistanceLoss(device = self.device)
        elif loss_name == self.LOSSES['multilabel_Minimum_Poincare']:
            self.selected_loss = multilabelMinimumPoincareDistanceLoss(device=self.device)
        elif loss_name == self.LOSSES['multilabel_Minimum_Normalized_Poincare']:
            self.selected_loss = multilabelMinimumNormalizedPoincareDistanceLoss(device=self.device)
        elif loss_name == self.LOSSES['multilabel_Minimum_Cosine']:
            self.selected_loss = multilabelMinimumCosineDistanceLoss(device=self.device)
        elif loss_name == self.LOSSES['multilabel_Average_Poincare']:
            self.selected_loss = multilabelAveragePoincareDistanceLoss(device=self.device)
        elif loss_name == self.LOSSES['multilabel_Average_Cosine']:
            self.selected_loss = multilabelAverageCosineDistanceLoss(device=self.device)
        
    def set_regularization_params(self, regul_dict):
        print('setting regul params for {}'.format(self.selected_loss))
        self.selected_loss.set_regularization(regul_dict)

    def compute_loss(self):
        loss = self.selected_loss.compute_loss(true = self.true_values,
                                                   pred = self.predictions)
        if not self.weighted:
            return loss
        else:
            batch_weights = self.get_batch_weights()
            return loss * batch_weights
    
    def compute_batch_weights(self):
        batch_weights = [self.weights[l.item()] for l in self.labels]
        self.batch_weights = torch.tensor(batch_weights, dtype=torch.float64, device = self.device)
    
    def get_batch_weights(self):
        return self.batch_weights
    
    def set_batch_weights(self, batch_weights):
        self.batch_weights = batch_weights


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

        frac = numerator/denom + 1

        acos = self.acosh(1  + frac)
        
        return acos


    def acosh(self, x):
        return torch.log(x + torch.sqrt(x**2-1))

class normalizedPoincareDistanceLoss(Loss):
    def compute_loss(self, true, pred):
        numerator = 2 * torch.norm(true - pred, dim = 1)**2

        pred_norm = torch.norm(pred, dim = 1)**2
        true_norm = torch.norm(true, dim = 1)**2

        left_denom = 1 - pred_norm
        right_denom = 1 - true_norm
        
        denom = left_denom * right_denom

        frac = numerator/denom + 1

        acos = self.acosh(1  + frac)
        
        return 1 - (1/(1 + acos))


    def acosh(self, x):
        return torch.log(x + torch.sqrt(x**2-1))

class regularizedPoincareDistanceLoss(poincareDistanceLoss):

    def set_regularization(self, regul):
        self.regul = regul
        print('regul setted')

    def mse(self, y_pred, y_true):    
        mse_loss = nn.MSELoss()
        return mse_loss(y_pred, y_true)

    def cosine_loss(self, true, pred):
        cossim = torch.nn.CosineSimilarity(dim = 1)
        return 1 - cossim(true, pred)

    def compute_loss(self, true, pred):
        acos = super().compute_loss(true = true, pred = pred)

        l0 = torch.tensor(1., device = self.device)
        l1 = torch.tensor(1., device = self.device)
        
        if sum(self.regul.values()) > 1:
            
            true_perm = true[torch.randperm(true.size()[0])]
            
            l0 = torch.abs(super().compute_loss(pred, true_perm) - super().compute_loss(true, true_perm))
            l1 = self.cosine_loss(pred, true)
        
        return acos**self.regul['distance_power'] + l0 * self.regul['negative_sampling'] + l1 * self.regul['mse']

class multilabelMinimumNormalizedPoincareDistanceLoss(normalizedPoincareDistanceLoss):
    def compute_loss(self, true, pred):
        # print('----------------------------')
        # print('true: {}'.format(true.shape))
        for i in range(true.shape[1]):
            t = torch.index_select(true, 1, index=torch.tensor([i], device=self.device)).squeeze()

            if len(t.shape) == 1:
                t = t.view(1, t.shape[0])

            # print('t: {}'.format(t.shape))
            # print('pred: {}'.format(pred.shape))
            
            loss = super().compute_loss(true = t, pred = pred).view(1, true.shape[0])
            
            if i == 0:
                min_tensor = loss
            else:
                catted = torch.cat((loss, min_tensor), dim = 0)
                min_tensor, _ = torch.min(catted, dim=0)
                min_tensor = min_tensor.view(1, true.shape[0])

        return min_tensor

class multilabelMinimumPoincareDistanceLoss(poincareDistanceLoss):
    def compute_loss(self, true, pred):
        # print('----------------------------')
        # print('true: {}'.format(true.shape))
        for i in range(true.shape[1]):
            t = torch.index_select(true, 1, index=torch.tensor([i], device=self.device)).squeeze()

            if len(t.shape) == 1:
                t = t.view(1, t.shape[0])

            # print('t: {}'.format(t.shape))
            # print('pred: {}'.format(pred.shape))
            
            loss = super().compute_loss(true = t, pred = pred).view(1, true.shape[0])
            
            if i == 0:
                min_tensor = loss
            else:
                catted = torch.cat((loss, min_tensor), dim = 0)
                min_tensor, _ = torch.min(catted, dim=0)
                min_tensor = min_tensor.view(1, true.shape[0])

        return min_tensor

class multilabelMinimumCosineDistanceLoss(cosineLoss):
    def compute_loss(self, true, pred):
        for i in range(true.shape[1]):
            t = torch.index_select(true, 1, index=torch.tensor([i], device=self.device)).squeeze()

            if len(t.shape) == 1:
                t = t.view(1, t.shape[0])

            # print('t: {}'.format(t.shape))
            # print('pred: {}'.format(pred.shape))
            
            loss = super().compute_loss(true = t, pred = pred).view(1, true.shape[0])
            
            if i == 0:
                min_tensor = loss
            else:
                catted = torch.cat((loss, min_tensor), dim = 0)
                min_tensor, _ = torch.min(catted, dim=0)
                min_tensor = min_tensor.view(1, true.shape[0])

        return min_tensor

class multilabelAveragePoincareDistanceLoss(poincareDistanceLoss):
    def compute_loss(self, true, pred):
        for i in range(true.shape[1]):
            t = torch.index_select(true, 1, index=torch.tensor([i], device=self.device)).squeeze()

            if len(t.shape) == 1:
                t = t.view(1, t.shape[0])

            # print('t: {}'.format(t.shape))
            # print('pred: {}'.format(pred.shape))
            
            loss = super().compute_loss(true = t, pred = pred).view(1, true.shape[0])
            if i == 0:
                mean_tensor = loss
            else:
                mean_tensor = torch.cat((loss, mean_tensor), dim = 0)


        return torch.mean(mean_tensor, dim = 0)

class multilabelAverageCosineDistanceLoss(cosineLoss):
    def compute_loss(self, true, pred):
        for i in range(true.shape[1]):
            t = torch.index_select(true, 1, index=torch.tensor([i], device=self.device)).squeeze()

            if len(t.shape) == 1:
                t = t.view(1, t.shape[0])

            # print('t: {}'.format(t.shape))
            # print('pred: {}'.format(pred.shape))
            
            loss = super().compute_loss(true = t, pred = pred).view(1, true.shape[0])
            
            if i == 0:
                mean_tensor = loss
            else:
                mean_tensor = torch.cat((loss, mean_tensor), dim = 0)

        return torch.mean(mean_tensor, dim = 0)

class personalBCELoss(Loss):
    def compute_loss(self, true, pred):
        pass
        

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

    def flashforward(self, input):
        return super().forward(input)

    def forward(self, input):

        input_vec = self.get_shimaoka_output(input)
        
        return super().forward(input_vec)
    
    def get_shimaoka_output(self, input):
        contexts, positions, context_len = input[0], input[1].double(), input[2]
        mentions, mention_chars = input[3], input[4]
        type_indexes = input[5]
                
        mention_vec = self.mention_encoder(mentions, mention_chars, self.word_lut)
        
        context_vec, attn = self.context_encoder(contexts, positions, context_len, self.word_lut)

        input_vec = torch.cat((mention_vec, context_vec), dim=1)

        return input_vec

    def type_prediction_on_test(self, topn, test_data, entities, labels):

        checkpoint = torch.load(self.checkpoint_path)

        try:
            print('loading model checkpoint at epoch {}'.format(checkpoint['epoch']))
        except:
            pass

        self.load_state_dict(checkpoint['model_state_dict'])

        self.eval()

        self.topn = topn

        x = [[], []]
        i = 0
        print('... extract prediction on test ...')
        for i in tqdm(range(len(test_data['data']))):
            t = test_data['data'][i]
            pred = self(t)
            x[0].extend(pred[0].detach().cpu().numpy())
            x[1].extend(pred[1].detach().cpu().numpy())
        
        self.compute_prediction(x, labels = labels, entities = entities)

class LopezLike(ShimaokaMTNCI):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.scaler = nn.Linear(750, 1, bias=True).cuda()
        self.sigmoid = nn.Sigmoid()


    def forward(self, input):
        contexts, positions, context_len = input[0], input[1].double(), input[2]
        mentions, mention_chars = input[3], input[4]
        type_indexes = input[5]
                
        mention_vec = self.mention_encoder(mentions, mention_chars, self.word_lut)
        context_vec, attn = self.context_encoder(contexts, positions, context_len, self.word_lut)

        input_vec = torch.cat((mention_vec, context_vec), dim=1)
        
        dis_pred, hyp_pred = super().flashforward(input_vec) 

        scalers = self.get_scalers(input_vec)
        direction_vectors = self.get_direction_vector(hyp_pred)

        return [dis_pred, direction_vectors * scalers]


    def get_direction_vector(self, input):
        norms = input.norm(p=2, dim=1, keepdim=True)
        return input.div(norms.expand_as(input))

    def get_scalers(self, input):
        output = self.scaler(input)
        return self.sigmoid(output)

class ChoiMTNCI(ShimaokaMTNCI):

    def compute_prediction(self, test_predictions, labels, entities = None):
        
        self.tsv = {'distributional':{'HITS': []},
                    'hyperbolic': {'HITS': []}
                    }

        for space, preds in zip(['distributional', 'hyperbolic'], test_predictions):
            print(' ...evaluating test predictions in {} space... '.format(space))
            self.emb = Embedding()            
            self.emb.set_embedding(self.datasetManager.concept_embeddings[space])
            self.emb.set_name(space)
            
            if space == 'distributional':
                self.emb.set_distance(cosine_dissimilarity)
                self.emb.set_centroid_method(vector_mean)
            elif space == 'hyperbolic':
                self.emb.set_distance(hyper_distance)
                self.emb.set_centroid_method(hyperbolic_midpoint)
            
            # print('occurrence {}'.format(space))
            # HITS = self.occurrence_level_prediction(predictions = preds, 
                                                    # labels = labels)
            
            # self.fill_TSV(space = space, HITS= HITS)

            # self.save_results(HITS=HITS,
                            #   level_name = 'Occurrence Level in {} space'.format(space))

            correct_granularities, all_granularities = self.compute_granularity_metric(predictions= preds, 
                                                                                        labels = labels)
            
            self.save_granularity_results(correct = correct_granularities, all_res = all_granularities)


            measures, perfect_measures = self.compute_IR_metrics(predictions = preds, labels_lists = labels)

            self.save_IR_results(measures = measures, perfect = perfect_measures)


    def occurrence_level_prediction(self, predictions, labels, granularities = ['coarse', 'fine', 'ultrafine', 'all']):

        corrects_n = 0
        total_n = len(labels)

        return self.compute_metrics(total=total_n,
                                    predictions= predictions,
                                    labels = labels,
                                    granularities=granularities)        

    def compute_metrics(self, total, 
                        predictions, labels, granularities = ['coarse', 'fine', 'ultrafine', 'all']):
        
        HITS_at_n = {t: 0 for t in self.topn}

        HITS_at_n = {g: copy.deepcopy(HITS_at_n) for g in granularities}

        concept_predicted = defaultdict(int)
        correct_prediction = defaultdict(int)
        corrects_n = {t: 0 for t in self.topn}

        corrects_n = {g: copy.deepcopy(corrects_n) for g in granularities}

        bar = tqdm(total=len(predictions))
        bar.set_description('getting results')
        
        all_ = {g: {t: 0 for t in self.topn} for g in granularities}

        i = 0

        for pred, label in zip(predictions, labels):
            for g in granularities:
                neigh = self.emb.get_neigh(vector = pred, top = max(self.topn), include = g)
                for t in self.topn:
                    new_neigh = neigh[:t]
                    filtered_labels = self.filter_labels(label, gran=g)
                    if filtered_labels:
                        all_[g][t] += 1
                        if set(filtered_labels).intersection(set(new_neigh)):
                            corrects_n[g][t] += 1
            bar.update(1)

            
            i += 1

        bar.close()

        for g in HITS_at_n.keys():
            for t in self.topn:
                HITS_at_n[g][t] = corrects_n[g][t]/all_[g][t]
        return HITS_at_n

    def save_results(self, HITS, level_name):
        with open(self.results_path, 'a') as out:
            out.write('\n{} results: \n'.format(level_name))
            for t in self.topn:
                for g in HITS.keys():
                    string = 'taking first {} {} neighbours'.format(t, g)         
                    out.write('\t HITS {:40}: {:.4f}\n'.format(string, HITS[g][t]))
                out.write('\n')

    def compute_granularity_metric(self, predictions, labels):
        correct_granularities = {'coarse': 0, 'fine': 0, 'ultrafine': 0}
        all_granularities = {'coarse': 0, 'fine': 0, 'ultrafine': 0}

        correct_granularities = {t: copy.deepcopy(correct_granularities) for t in self.topn}
        all_granularities = {t: copy.deepcopy(all_granularities) for t in self.topn}

        bar = tqdm(total=len(predictions))

        i = 0
        for pred, label in zip(predictions, labels):
            bar.set_description('getting results granularity distributions')
            neigh = self.emb.get_neigh(vector = pred, top = max(self.topn))
        
            for t in self.topn:
                new_neigh = neigh[:t]
                for n in new_neigh:
                    if n in COARSE:
                        all_granularities[t]['coarse'] += 1
                        if n in label:
                            correct_granularities[t]['coarse'] += 1
                    elif n in FINE:
                        all_granularities[t]['fine'] += 1
                        if n in label:
                            correct_granularities[t]['fine'] += 1
                    else:
                        all_granularities[t]['ultrafine'] += 1
                        if n in label:
                            correct_granularities[t]['ultrafine'] += 1
            bar.update(1)

            # if i > 9:
            #     break

            i += 1

        bar.close()
        
        return correct_granularities, all_granularities

    def save_TSV_granularity(self, correct, all_res):
        with open(self.TSV_path, 'a') as out:
            out.write('\nresults for {}\n'.format(self.emb.name))
            out.write('granularity:\n')
            for t in self.topn:

                sum_all = sum(all_res[t].values())
                sum_correct = sum(correct[t].values())

                coarse_frac = 0
                fine_frac = 0
                ultrafine_frac = 0

                if sum_correct:
                    coarse_frac = correct[t]['coarse']/sum_correct
                    fine_frac = correct[t]['fine']/sum_correct
                    ultrafine_frac = correct[t]['ultrafine']/sum_correct
                string = ''

                string += '{:.4f}\t'.format(coarse_frac)
                string += '{:.4f}\t'.format(fine_frac)
                string += '{:.4f}\t'.format(ultrafine_frac)

                string += '{}\t'.format(sum_correct)

                correct_coarse = all_res[t]['coarse']/sum_all
                correct_fine = all_res[t]['fine']/sum_all
                correct_ultra = all_res[t]['ultrafine']/sum_all

                string += '{:.4f}\t'.format(correct_coarse)
                string += '{:.4f}\t'.format(correct_fine)
                string += '{:.4f}\t\n'.format(correct_ultra)

                out.write(string)


    def save_granularity_results(self, correct, all_res):

        self.save_TSV_granularity(correct, all_res)

        with open(self.results_path, 'a') as out:
            out.write('\nGranularity Metrics:')
            for t in self.topn:
                out.write('\n------------------------ topn: {}--------------------------\n'.format(t))


                sum_all = sum(all_res[t].values())
                sum_correct = sum(correct[t].values())

                coarse_frac = 0
                fine_frac = 0
                ultrafine_frac = 0

                if sum_correct:
                    coarse_frac = correct[t]['coarse']/sum_correct
                    fine_frac = correct[t]['fine']/sum_correct
                    ultrafine_frac = correct[t]['ultrafine']/sum_correct

                out.write('\t all predictions: {}, correct predictions: {}\n'.format(sum_all, sum_correct))
                out.write('\t predicted coarse: {:.2f}\n\t predicted fine: {:.2f}\n\t predicted ultrafine: {:.2f}\n\n'.format(all_res[t]['coarse']/sum_all,
                                                                                                                          all_res[t]['fine']/sum_all,
                                                                                                                          all_res[t]['ultrafine']/sum_all))
                out.write('\t correct coarse: {:.2f}\n\t correct fine: {:.2f}\n\t correct ultrafine: {:.2f}\n'.format(coarse_frac ,
                                                                                                                    fine_frac,
                                                                                                                    ultrafine_frac))




    def compute_IR_metrics(self, predictions, labels_lists, granularities = ['coarse', 'fine', 'ultrafine', 'all']):
        
        precision = {g: {t: 0 for t in self.topn} for g in granularities}
        recall = {g: {t: 0 for t in self.topn} for g in granularities}        
        
        precision_values = {g: {t: [] for t in self.topn} for g in granularities}
        recall_values = {g: {t: [] for t in self.topn} for g in granularities}
        
        perfect_score_precision = {g: {t: [] for t in self.topn} for g in granularities}
        perfect_score_recall = {g: {t: [] for t in self.topn} for g in granularities}
        perfect_score_fmeasure = {g: {t: [] for t in self.topn} for g in granularities}

        fmeasure = {g: {t: 0 for t in self.topn} for g in granularities}

        bar = tqdm(total=len(predictions))

        i = 0
        for pred, labels in zip(predictions, labels_lists):
            bar.set_description('getting IR results')
            for g in granularities:
                neigh = self.emb.get_neigh(vector = pred, top = max(self.topn), include=g)
                
                if g != 'all':
                    filtered_labels = self.filter_labels(labels=labels, gran = g)
                else:
                    filtered_labels = labels
                if filtered_labels:
                    for t in self.topn:
                        new_neigh = neigh[:t]
                        corrects = 0
                        for n in new_neigh:
                            if n in filtered_labels:
                                corrects += 1
                        prec = corrects / len(new_neigh)
                        rec = corrects / len(filtered_labels)

                        if len(filtered_labels) > t:
                            perfect_score_precision[g][t].append(1)
                            perfect_score_recall[g][t].append(t/len(filtered_labels))
                        else:
                            perfect_score_precision[g][t].append(len(filtered_labels)/t)
                            perfect_score_recall[g][t].append(1)

                        precision_values[g][t].append(prec)
                        recall_values[g][t].append(rec)
            bar.update(1)

            # if i > 9:
            #     break

            i += 1
        
        bar.close()
        for g in granularities:
            for t in self.topn:
                precision[g][t] = np.mean(precision_values[g][t])
                recall[g][t] = np.mean(recall_values[g][t])
                fmeasure[g][t] = self.f1(p=precision[g][t], r=recall[g][t])

                perfect_score_precision[g][t] = np.mean(perfect_score_precision[g][t])
                perfect_score_recall[g][t] = np.mean(perfect_score_recall[g][t])
                perfect_score_fmeasure[g][t] = self.f1(p=perfect_score_precision[g][t], r=perfect_score_recall[g][t])

        return [precision, recall, fmeasure], [perfect_score_precision, perfect_score_recall, perfect_score_fmeasure]

    def filter_labels(self, labels, gran):
        if gran == 'coarse':
            filter_list = COARSE
        elif gran == 'fine':
            filter_list = FINE
        else:
            filter_list = set(self.emb.embedding.keys()).difference(set(COARSE))
            filter_list = filter_list.difference(set(FINE))

        return [l for l in labels if l in filter_list]

    def save_TSV_IR(self, measures, perfect):

        with open(self.TSV_path, 'a') as out:

            lopez_lines = ['', '', '']
            # coarse_lines = ['', '', '']
            # fine_lines = ['', '', '', '']
            # ultrafine_lines = ['', '', '']
            # all_lines = ['', '', '']

            lines = ['', '', '']

            k_dict = {'coarse': 1, 'fine': 1, 'ultrafine': 3, 'all': 5}

            out.write('\nlopez-like evaluation:\n')

            for g in list(measures[0].keys()):
                t = k_dict[g]
                i = 0
                for l, m, d_m in zip(lopez_lines, ['precision', 'recall', 'fmeasure'], measures):
                    lopez_lines[i] = l + '{:.4f}\t'.format(d_m[g][t])
                    i += 1
            
            for l in lopez_lines:
                out.write(l[:-2] + '\n')
            

            out.write('\n single granularity metrics: \n'.format(g))
            
            granularities = ['coarse', 'fine', 'ultrafine', 'all']

            for g in granularities:
                for t in self.topn:
                    i = 0
                    for l, m, d_m in zip(lines, ['precision', 'recall', 'fmeasure'], measures):
                        if g in list(measures[0].keys()):
                            lines[i] = l + '{:.4f}\t'.format(d_m[g][t])
                        else:
                            lines[i] = l + '\t'
                        i += 1

            for l in lines:
                out.write(l[:-2] + '\n')


    def save_IR_results(self, measures, perfect):

        self.save_TSV_IR(measures, perfect)

        with open(self.results_path, 'a') as out:
            out.write('IR Results\n')
            string = '\t{:10}: {:4.2f} | perfect {:10}: {:4.2f}\n'
            granularities = list(measures[0].keys())
            for t in self.topn:
                out.write('\n------------------------ topn: {}--------------------------\n'.format(t))
                for g in granularities:
                    out.write('---------------- take first {} {} neighbours ----------------\n'.format(t, g))
                    for m, d_m, d_p in zip(['precision', 'recall', 'fmeasure'], measures, perfect):
                        out.write(string.format(m, d_m[g][t], m, d_p[g][t]))
                    out.write('\n')
                    

    def fill_TSV(self, space, HITS):
        self.tsv[space]['HITS'].append(HITS)

    def compute_measures_and_perfect(self, prediction, true_labels):
        
        measures = {'precision': [], 'recall': [], 'fmeasure': []} 
        perfect_measures = {'precision': [], 'recall': [], 'fmeasure': []}

        for p, t in zip(prediction, true_labels):
            p = set(p)
            t = set(t)

            correct_pred = p.intersection(t)

            if correct_pred:
                measures['precision'].append(len(correct_pred)/len(p))
                measures['recall'].append(len(correct_pred)/len(t))
                measures['fmeasure'].append(self.f1(measures['precision'][-1], measures['recall'][-1]))
            else:
                measures['precision'].append(0)
                measures['recall'].append(0)
                measures['fmeasure'].append(0)

            p_precision = len(t)/len(p)
            if p_precision > 1:
                p_precision = 1
            
            if len(t) <= len(p):
                p_recall = 1
            else:
                p_recall = len(p)/len(t)

            perfect_measures['precision'].append(p_precision)
            perfect_measures['recall'].append(p_recall)
            perfect_measures['fmeasure'].append(self.f1(p_precision, p_recall))

        return measures, perfect_measures 

class SingleClassifierMTNCI(ShimaokaMTNCI):
     
    def __init__(self, class_number, *args, **kwargs):
        
        super().__init__(*args, **kwargs)

        self.classifierLayer_1 =  nn.Linear(kwargs['dims'][-1] + 20, 256, bias=True).cuda()
        # self.classifierLayer_1 =  nn.Linear(20, 256, bias=True).cuda()
        self.classifierLayer_2 =  nn.Linear(256, class_number,bias=True).cuda()
        self.sigmoid = nn.Sigmoid()
        self.leaky_relu = nn.LeakyReLU(0.1).cuda()
        self.classification_loss = nn.BCELoss() 
        self.dropout = nn.Dropout(p=0.3).cuda()



    def forward(self, input):

        regression_vectors = super().forward(input)

        shimaoka_vector = self.get_shimaoka_output(input)
        common_output = self.common_network(shimaoka_vector)

        classifier_input = torch.cat((regression_vectors[0], regression_vectors[1], common_output), dim = 1)
        # classifier_input = torch.cat((regression_vectors[0], regression_vectors[1]), dim = 1)

        classifier_output = self.dropout(self.leaky_relu(self.classifierLayer_1(classifier_input)))
        classifier_output = self.sigmoid(self.classifierLayer_2(classifier_output))

        return regression_vectors, classifier_output

    def train_model(self):

        train_length = len(self.datasetManager.Y_train)
        val_length = len(self.datasetManager.Y_val)

        losses = Prediction(device = self.device).LOSSES

        distributional_prediction_train_manager = self.get_prediction_manager(loss_name=self.losses_name_dict['distributional'])
        distributional_prediction_val_manager = self.get_prediction_manager(loss_name=self.losses_name_dict['distributional'])

        hyperbolic_prediction_train_manager = self.get_prediction_manager(loss_name=self.losses_name_dict['hyperbolic-train'])
        
        hyperbolic_prediction_val_manager = self.get_prediction_manager(loss_name=self.losses_name_dict['hyperbolic-val'])

        hyperbolic_train_metric_manager = self.get_prediction_manager(loss_name=self.metric_name_dict['hyperbolic'])
        distributional_train_metric_manager = self.get_prediction_manager(loss_name=self.metric_name_dict['distributional'])
        
        hyperbolic_val_metric_manager = self.get_prediction_manager(loss_name=self.metric_name_dict['hyperbolic'])
        distributional_val_metric_manager = self.get_prediction_manager(loss_name=self.metric_name_dict['distributional'])

        if self.weighted:

            train_weights = self.datasetManager.get_weights(label = 'Train')
            val_weights = self.datasetManager.get_weights(label = 'Val')

            distributional_prediction_train_manager.set_weight(train_weights)
            hyperbolic_prediction_train_manager.set_weight(train_weights)

            distributional_prediction_val_manager.set_weight(val_weights)
            hyperbolic_prediction_val_manager.set_weight(val_weights)

            hyperbolic_train_metric_manager.set_weight(train_weights)
            distributional_train_metric_manager.set_weight(train_weights)

            hyperbolic_val_metric_manager.set_weight(val_weights)
            distributional_val_metric_manager.set_weight(val_weights)

        for epoch in range(self.epochs):
            
            self.datasetManager.get_epoch_data(initialize=True)

            train_loss_SUM = 0
            val_loss_SUM = 0
            
            distributional_train_loss_SUM = 0
            distributional_val_loss_SUM = 0
            
            hyperbolic_train_loss_SUM = 0
            hyperbolic_val_loss_SUM = 0

            distributional_train_metric_SUM = 0
            hyperbolic_train_metric_SUM = 0
            
            distributional_val_metric_SUM = 0
            hyperbolic_val_metric_SUM = 0
            
            train_weights_sum = 0
            val_weights_sum = 0

            train_length = 0

            bar = tqdm(total=self.TIMES * self.datasetManager.get_data_batch_length(dataset='train'))
            for _ in range(self.TIMES):
                for batch_iteration in range(self.datasetManager.get_data_batch_length(dataset = 'train')):

                    if random.random() < self.PERC:

                        x, labels, targets, numeric_labels = self.datasetManager.get_epoch_data(dataset='train', batch_iteration = batch_iteration)

                        # print('labels: {}'.format(labels))
                        # print('targets: {}'.format(targets['distributional']))
                        # print('numeric_labels: {}'.format(numeric_labels))
                        ######################
                        ####### TRAIN ########
                        ######################
                        # print('xs: {}'.format(x[0].shape))
                        # print('targets: {}'.format(targets['distributional'].shape))
                        self.optimizer.zero_grad()
                        
                        self.train()
                        
                        output, classifier_output = self(x)

                        # print('output: {}'.format(output))
                        # print('classifier_output: {}'.format(classifier_output))

                        distributional_prediction_train_manager.set_prediction(predictions = output[0],
                                                                        true_values = targets['distributional'].squeeze(), 
                                                                        labels = labels)

                        hyperbolic_prediction_train_manager.set_prediction(predictions = output[1], 
                                                                    true_values = targets['hyperbolic'].squeeze(),
                                                                    labels = labels) 

                        distributional_train_metric_manager.set_prediction(predictions = output[0],
                                                                                        true_values = targets['distributional'].squeeze(), 
                                                                                        labels = labels)

                        hyperbolic_train_metric_manager.set_prediction(predictions = output[1], 
                                                                    true_values = targets['hyperbolic'].squeeze(),
                                                                    labels = labels)                

                        if self.weighted:
                            distributional_prediction_train_manager.compute_batch_weights()
                            batch_weights = distributional_prediction_train_manager.get_batch_weights()
                            train_weights_sum += torch.sum(batch_weights)

                            distributional_prediction_train_manager.set_batch_weights(batch_weights)
                            hyperbolic_prediction_train_manager.set_batch_weights(batch_weights)
                            distributional_train_metric_manager.set_batch_weights(batch_weights)
                            hyperbolic_train_metric_manager.set_batch_weights(batch_weights)

                        distributional_train_loss = distributional_prediction_train_manager.compute_loss()
                        
                        hyperbolic_train_loss = hyperbolic_prediction_train_manager.compute_loss()
                        
                        distributional_train_loss_SUM += torch.sum(distributional_train_loss * self.llambdas['distributional']).item()
                        hyperbolic_train_loss_SUM += torch.sum(hyperbolic_train_loss * self.llambdas['hyperbolic']).item()
                        

                        distributional_train_metric = distributional_train_metric_manager.compute_loss()
                        hyperbolic_train_metric = hyperbolic_train_metric_manager.compute_loss()

                        distributional_train_metric_SUM += torch.sum(distributional_train_metric).item()
                        hyperbolic_train_metric_SUM += torch.sum(hyperbolic_train_metric).item()

                        train_loss = self.get_multitask_loss({'distributional': distributional_train_loss, 
                                                            'hyperbolic': hyperbolic_train_loss})
                        
                        classifier_loss = self.classification_loss(classifier_output, numeric_labels) * self.beta

                        train_loss = train_loss + classifier_loss 

                        train_loss_SUM += train_loss.item()

                        train_loss.backward()
                        self.optimizer.step()

                        train_length += len(labels)

                    bar.update(1)
            bar.close()

            with torch.no_grad():
                self.eval()   
                classifier_loss_sum = 0
                bar_val = tqdm(total=self.datasetManager.get_data_batch_length(dataset='val'))
                # for batch_iteration in range(len(self.datasetManager.valloader)):
                for batch_iteration in range(self.datasetManager.get_data_batch_length(dataset='val')):
                    x, labels, targets, numeric_labels = self.datasetManager.get_epoch_data(dataset='val', batch_iteration = batch_iteration)

                    ######################
                    ######## VAL #########
                    ######################

                    output, classifier_output = self(x)

                    distributional_prediction_val_manager.set_prediction(predictions = output[0],
                                                                    true_values = targets['distributional'].squeeze(),
                                                                    labels = labels)

                    hyperbolic_prediction_val_manager.set_prediction(predictions = output[1], 
                                                                true_values = targets['hyperbolic'].squeeze(),
                                                                labels = labels) 
                    
                    distributional_val_metric_manager.set_prediction(predictions = output[0],
                                                                            true_values = targets['distributional'].squeeze(), 
                                                                            labels = labels)

                    hyperbolic_val_metric_manager.set_prediction(predictions = output[1], 
                                                                        true_values = targets['hyperbolic'].squeeze(),
                                                                        labels = labels)                


                    if self.weighted:
                        distributional_prediction_val_manager.compute_batch_weights()
                        batch_weights = distributional_prediction_val_manager.get_batch_weights()
                        val_weights_sum += torch.sum(batch_weights)

                        distributional_prediction_val_manager.set_batch_weights(batch_weights)
                        hyperbolic_prediction_val_manager.set_batch_weights(batch_weights)
                        distributional_val_metric_manager.set_batch_weights(batch_weights)
                        hyperbolic_val_metric_manager.set_batch_weights(batch_weights)


                    distributional_val_loss = distributional_prediction_val_manager.compute_loss()
                    hyperbolic_val_loss = hyperbolic_prediction_val_manager.compute_loss()

                    distributional_val_loss_SUM += torch.sum(distributional_val_loss * self.llambdas['distributional']).item()
                    hyperbolic_val_loss_SUM += torch.sum(hyperbolic_val_loss * (self.llambdas['hyperbolic'])).item()
                    
                    distributional_val_metric = distributional_val_metric_manager.compute_loss()
                    hyperbolic_val_metric = hyperbolic_val_metric_manager.compute_loss()

                    distributional_val_metric_SUM += torch.sum(distributional_val_metric).item()
                    hyperbolic_val_metric_SUM += torch.sum(hyperbolic_val_metric).item()

                    val_loss = self.get_multitask_loss({'distributional': distributional_val_loss, 
                                                        'hyperbolic': hyperbolic_val_loss})
                    

                    classifier_loss = self.classification_loss(classifier_output, numeric_labels) * self.beta

                    classifier_loss_sum += classifier_loss.item()

                    val_loss = val_loss + classifier_loss 

                    val_loss_SUM += val_loss.item()

                    bar_val.update(1)

            bar_val.close()
            if not self.weighted:
                train_loss_value = train_loss_SUM/train_length
                val_loss_value = val_loss_SUM/val_length

                hyperbolic_train_loss_value = hyperbolic_train_loss_SUM/train_length
                hyperbolic_val_loss_value = hyperbolic_val_loss_SUM/val_length

                distributional_train_loss_value = distributional_train_loss_SUM/train_length
                distributional_val_loss_value = distributional_val_loss_SUM/val_length

                distributional_train_metric_value = distributional_train_metric_SUM/train_length
                hyperbolic_train_metric_value = hyperbolic_train_metric_SUM/train_length

                distributional_val_metric_value = distributional_val_metric_SUM/val_length
                hyperbolic_val_metric_value = hyperbolic_val_metric_SUM/val_length

                classifier_loss_value = classifier_loss_sum/val_length

            else:
                train_loss_value = train_loss_SUM/train_weights_sum
                val_loss_value = val_loss_SUM/val_weights_sum

                hyperbolic_train_loss_value = hyperbolic_train_loss_SUM/train_weights_sum
                hyperbolic_val_loss_value = hyperbolic_val_loss_SUM/val_weights_sum

                distributional_train_loss_value = distributional_train_loss_SUM/train_weights_sum
                distributional_val_loss_value = distributional_val_loss_SUM/val_weights_sum

                distributional_train_metric_value = distributional_train_metric_SUM/train_weights_sum
                hyperbolic_train_metric_value = hyperbolic_train_metric_SUM/train_weights_sum

                distributional_val_metric_value = distributional_val_metric_SUM/val_weights_sum
                hyperbolic_val_metric_value = hyperbolic_val_metric_SUM/val_weights_sum
                

            self.checkpointManager(val_loss_value, epoch)

            if self.early_stopping:
                break

            losses_dict = {'Losses/Hyperbolic Losses': {'Train': hyperbolic_train_loss_value, 
                                                 'Val': hyperbolic_val_loss_value},
                           'Losses/Distributional Losses':  {'Train': distributional_train_loss_value,
                                                      'Val': distributional_val_loss_value},
                           'Losses/MTL-Losses': {'Train': train_loss_value,
                                          'Val': val_loss_value}
            }


            metric_dict = {'Metrics/Hyperbolic Metrics': {'Train': hyperbolic_train_metric_value, 
                                                 'Val': hyperbolic_val_metric_value},
                           'Metrics/Distributional Metrics':  {'Train': distributional_train_metric_value,
                                                                'Val': distributional_val_metric_value}
                            }
            self.log_losses(losses_dict = losses_dict, epoch = epoch + 1)

            self.log_losses(losses_dict=metric_dict, epoch = epoch + 1)

            print('{:^25}'.format('epoch {:^3}/{:^3}'.format(epoch, self.epochs)))
            print('{:^25}'.format('Train loss: {:.4f}, Val loss: {:.4f}, Min loss: {:.4f} at epoch: {}'.format(train_loss_value, 
                                                                                                    val_loss_value, 
                                                                                                    self.min_loss, 
                                                                                                    self.best_epoch)))
            print('{:^25}'.format('T_MHD: {:.4f}, V_MHD:{:.4f}'.format(hyperbolic_train_metric_value, hyperbolic_val_metric_value)))
            print('{:^25}'.format('T_MDD: {:.4f}, V_MDD:{:.4f}'.format(distributional_train_metric_value, distributional_val_metric_value)))
            print('classifier_loss: {:.4f}'.format(classifier_loss_value))


    def type_prediction_on_test(self, topn, test_data, entities, labels):

        checkpoint = torch.load(self.checkpoint_path)

        try:
            print('loading model checkpoint at epoch {}'.format(checkpoint['epoch']))
            with open(self.TSV_path, 'a') as out:
                out.write('-------------------------------------')
                out.write('best epoch: {}'.format(checkpoint['epoch']))
                out.write('-------------------------------------\n')


            with open(self.results_path, 'a') as out:
                out.write('-------------------------------------')
                out.write('best epoch: {}'.format(checkpoint['epoch']))
                out.write('-------------------------------------\n')
        except:
            pass

        self.load_state_dict(checkpoint['model_state_dict'])

        self.eval()

        self.topn = [1]

        x = [[], []]
        class_preds = []
        i = 0
        print('... extract prediction on test ...')
        for i in tqdm(range(len(test_data['data']))):
            t = test_data['data'][i]
            pred, class_pred = self(t)
            x[0].extend(pred[0].detach().cpu().numpy())
            x[1].extend(pred[1].detach().cpu().numpy())
            class_preds.extend(class_pred.detach().cpu().numpy())

        self.emb = Embedding()           
        self.emb.set_embedding(self.datasetManager.concept_embeddings['hyperbolic'])
        self.emb.set_name('hyperbolic')

        labels = [l[0] for l in labels]
            
        self.compute_classifier_prediction(class_preds, labels)
        
        super().compute_prediction(x, labels = labels, entities = entities)

    def extract_top(self, class_preds, t):
        
        indexes = class_preds.argsort()[-t:][::-1]

        return self.datasetManager.get_label_from_numeric([indexes])
    
    def compute_classifier_prediction(self, class_preds, labels):
        t = self.topn[-1]
        
        top_preds = []
        for c in class_preds:
            pred = self.extract_top(c, t)
            top_preds.append(pred)

        self.tsv = {'classifier':{'recalls': [],
                                'precisions': [],
                                'fmeasures': []},
                    }


        concept_number = {s:len([l for l in labels if l == s]) for s in set(labels)}

        recalls, precisions, fmeasures = self.compute_classifier_metrics(total = len(top_preds), 
                                                                        predictions = top_preds, 
                                                                        labels = labels,
                                                                        elements_number_for_concept = concept_number)

        
        self.fill_TSV(space = 'classifier', precisions=precisions, recalls=recalls, fmeasures=fmeasures)
        self.save_results(recalls = recalls, 
                            precisions = precisions, 
                            fmeasures = fmeasures, 
                            level_name = 'Occurrence Level in {} space'.format('classifier'))


        

    def compute_classifier_metrics(self, total, predictions, labels, elements_number_for_concept):
            
            concept_accuracies = {t: {s:0 for s in set(labels)} for t in self.topn}

            precision_at_n = {t: 0 for t in self.topn}
            recall_at_n = {t: 0 for t in self.topn}
            f_measure = {t: 0 for t in self.topn}

            concept_predicted = defaultdict(int)
            correct_prediction = defaultdict(int)
            corrects_n = {t: 0 for t in self.topn}

            bar = tqdm(total=len(predictions))
            bar.set_description('getting results')
            
            i = 0

            for pred, label in zip(predictions, labels):
                for t in self.topn:
                    new_neigh = [p[0] for p in pred[:t]]
                    if label in new_neigh:
                        corrects_n[t] += 1
                        concept_accuracies[t][label] += 1
                
                    if t == 1:
                        concept_predicted[new_neigh[0]] += 1
                        if label in new_neigh:
                            correct_prediction[new_neigh[0]] += 1
                    
                    i += 1

            bar.close()

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


class ClassifierMTNCI(ChoiMTNCI):
     
    def __init__(self, class_number, *args, **kwargs):
        
        super().__init__(*args, **kwargs)

        self.classifierLayer_1 =  nn.Linear(kwargs['dims'][-1] + 20, 256, bias=True).cuda()
        # self.classifierLayer_1 =  nn.Linear(20, 256, bias=True).cuda()
        self.classifierLayer_2 =  nn.Linear(256, class_number,bias=True).cuda()
        self.sigmoid = nn.Sigmoid()
        self.leaky_relu = nn.LeakyReLU(0.1).cuda()
        self.classification_loss = nn.BCELoss() 


    def forward(self, input):

        regression_vectors = super().forward(input)

        shimaoka_vector = self.get_shimaoka_output(input)
        # common_output = self.common_network(shimaoka_vector)

        # classifier_input = torch.cat((regression_vectors[0], regression_vectors[1], common_output), dim = 1)
        classifier_input = torch.cat((regression_vectors[0], regression_vectors[1]), dim = 1)

        classifier_output = self.leaky_relu(self.classifierLayer_1(classifier_input))
        classifier_output = self.sigmoid(self.classifierLayer_2(classifier_output))

        return regression_vectors, classifier_output

    def train_model(self):

        train_length = len(self.datasetManager.Y_train)
        val_length = len(self.datasetManager.Y_val)

        losses = Prediction(device = self.device).LOSSES

        distributional_prediction_train_manager = self.get_prediction_manager(loss_name=self.losses_name_dict['distributional'])
        distributional_prediction_val_manager = self.get_prediction_manager(loss_name=self.losses_name_dict['distributional'])

        hyperbolic_prediction_train_manager = self.get_prediction_manager(loss_name=self.losses_name_dict['hyperbolic-train'])
        
        hyperbolic_prediction_val_manager = self.get_prediction_manager(loss_name=self.losses_name_dict['hyperbolic-val'])

        hyperbolic_train_metric_manager = self.get_prediction_manager(loss_name=self.metric_name_dict['hyperbolic'])
        distributional_train_metric_manager = self.get_prediction_manager(loss_name=self.metric_name_dict['distributional'])
        
        hyperbolic_val_metric_manager = self.get_prediction_manager(loss_name=self.metric_name_dict['hyperbolic'])
        distributional_val_metric_manager = self.get_prediction_manager(loss_name=self.metric_name_dict['distributional'])

        if self.weighted:

            train_weights = self.datasetManager.get_weights(label = 'Train')
            val_weights = self.datasetManager.get_weights(label = 'Val')

            distributional_prediction_train_manager.set_weight(train_weights)
            hyperbolic_prediction_train_manager.set_weight(train_weights)

            distributional_prediction_val_manager.set_weight(val_weights)
            hyperbolic_prediction_val_manager.set_weight(val_weights)

            hyperbolic_train_metric_manager.set_weight(train_weights)
            distributional_train_metric_manager.set_weight(train_weights)

            hyperbolic_val_metric_manager.set_weight(val_weights)
            distributional_val_metric_manager.set_weight(val_weights)

        for epoch in range(self.epochs):
            
            self.datasetManager.get_epoch_data(initialize=True)

            train_loss_SUM = 0
            val_loss_SUM = 0
            
            distributional_train_loss_SUM = 0
            distributional_val_loss_SUM = 0
            
            hyperbolic_train_loss_SUM = 0
            hyperbolic_val_loss_SUM = 0

            distributional_train_metric_SUM = 0
            hyperbolic_train_metric_SUM = 0
            
            distributional_val_metric_SUM = 0
            hyperbolic_val_metric_SUM = 0
            
            train_weights_sum = 0
            val_weights_sum = 0

            bar = tqdm(total=self.TIMES * self.datasetManager.get_data_batch_length(dataset='train'))
            for _ in range(self.TIMES):
                for batch_iteration in range(self.datasetManager.get_data_batch_length(dataset = 'train')):

                    if random.random() < self.PERC:

                        x, labels, targets, numeric_labels = self.datasetManager.get_epoch_data(dataset='train', batch_iteration = batch_iteration)

                        ######################
                        ####### TRAIN ########
                        ######################
                        # print('xs: {}'.format(x[0].shape))
                        # print('targets: {}'.format(targets['hyperbolic'].shape))
                        self.optimizer.zero_grad()
                        
                        self.train()
                        
                        output, classifier_output = self(x)

                        distributional_prediction_train_manager.set_prediction(predictions = output[0],
                                                                        true_values = targets['distributional'], 
                                                                        labels = labels)

                        hyperbolic_prediction_train_manager.set_prediction(predictions = output[1], 
                                                                    true_values = targets['hyperbolic'],
                                                                    labels = labels) 

                        distributional_train_metric_manager.set_prediction(predictions = output[0],
                                                                                        true_values = targets['distributional'], 
                                                                                        labels = labels)

                        hyperbolic_train_metric_manager.set_prediction(predictions = output[1], 
                                                                    true_values = targets['hyperbolic'],
                                                                    labels = labels)                

                        if self.weighted:
                            distributional_prediction_train_manager.compute_batch_weights()
                            batch_weights = distributional_prediction_train_manager.get_batch_weights()
                            train_weights_sum += torch.sum(batch_weights)

                            distributional_prediction_train_manager.set_batch_weights(batch_weights)
                            hyperbolic_prediction_train_manager.set_batch_weights(batch_weights)
                            distributional_train_metric_manager.set_batch_weights(batch_weights)
                            hyperbolic_train_metric_manager.set_batch_weights(batch_weights)

                        distributional_train_loss = distributional_prediction_train_manager.compute_loss()
                        
                        hyperbolic_train_loss = hyperbolic_prediction_train_manager.compute_loss()
                        
                        distributional_train_loss_SUM += torch.sum(distributional_train_loss * self.llambdas['distributional']).item()
                        hyperbolic_train_loss_SUM += torch.sum(hyperbolic_train_loss * self.llambdas['hyperbolic']).item()
                        

                        distributional_train_metric = distributional_train_metric_manager.compute_loss()
                        hyperbolic_train_metric = hyperbolic_train_metric_manager.compute_loss()

                        distributional_train_metric_SUM += torch.sum(distributional_train_metric).item()
                        hyperbolic_train_metric_SUM += torch.sum(hyperbolic_train_metric).item()

                        train_loss = self.get_multitask_loss({'distributional': distributional_train_loss, 
                                                            'hyperbolic': hyperbolic_train_loss})
                        

                        classifier_loss = self.classification_loss(classifier_output, numeric_labels) * 500

                        train_loss = train_loss + classifier_loss 

                        train_loss_SUM += train_loss.item()

                        train_loss.backward()
                        self.optimizer.step()
                    bar.update(1)
            bar.close()

            with torch.no_grad():
                self.eval()   
                classifier_loss_sum = 0
                bar_val = tqdm(total=self.datasetManager.get_data_batch_length(dataset='val'))
                # for batch_iteration in range(len(self.datasetManager.valloader)):
                for batch_iteration in range(self.datasetManager.get_data_batch_length(dataset='val')):
                    x, labels, targets, numeric_labels = self.datasetManager.get_epoch_data(dataset='val', batch_iteration = batch_iteration)

                    ######################
                    ######## VAL #########
                    ######################

                    output, classifier_output = self(x)

                    distributional_prediction_val_manager.set_prediction(predictions = output[0],
                                                                    true_values = targets['distributional'],
                                                                    labels = labels)

                    hyperbolic_prediction_val_manager.set_prediction(predictions = output[1], 
                                                                true_values = targets['hyperbolic'],
                                                                labels = labels) 
                    
                    distributional_val_metric_manager.set_prediction(predictions = output[0],
                                                                            true_values = targets['distributional'], 
                                                                            labels = labels)

                    hyperbolic_val_metric_manager.set_prediction(predictions = output[1], 
                                                                        true_values = targets['hyperbolic'],
                                                                        labels = labels)                


                    if self.weighted:
                        distributional_prediction_val_manager.compute_batch_weights()
                        batch_weights = distributional_prediction_val_manager.get_batch_weights()
                        val_weights_sum += torch.sum(batch_weights)

                        distributional_prediction_val_manager.set_batch_weights(batch_weights)
                        hyperbolic_prediction_val_manager.set_batch_weights(batch_weights)
                        distributional_val_metric_manager.set_batch_weights(batch_weights)
                        hyperbolic_val_metric_manager.set_batch_weights(batch_weights)


                    distributional_val_loss = distributional_prediction_val_manager.compute_loss()
                    hyperbolic_val_loss = hyperbolic_prediction_val_manager.compute_loss()

                    distributional_val_loss_SUM += torch.sum(distributional_val_loss * self.llambdas['distributional']).item()
                    hyperbolic_val_loss_SUM += torch.sum(hyperbolic_val_loss * (self.llambdas['hyperbolic'])).item()
                    
                    distributional_val_metric = distributional_val_metric_manager.compute_loss()
                    hyperbolic_val_metric = hyperbolic_val_metric_manager.compute_loss()

                    distributional_val_metric_SUM += torch.sum(distributional_val_metric).item()
                    hyperbolic_val_metric_SUM += torch.sum(hyperbolic_val_metric).item()

                    val_loss = self.get_multitask_loss({'distributional': distributional_val_loss, 
                                                        'hyperbolic': hyperbolic_val_loss})
                    

                    classifier_loss = self.classification_loss(classifier_output, numeric_labels) * 500

                    classifier_loss_sum += classifier_loss.item()


                    val_loss = val_loss + classifier_loss 

                    val_loss_SUM += val_loss.item()

                    bar_val.update(1)

            bar_val.close()
            if not self.weighted:
                train_loss_value = train_loss_SUM/train_length
                val_loss_value = val_loss_SUM/val_length

                hyperbolic_train_loss_value = hyperbolic_train_loss_SUM/train_length
                hyperbolic_val_loss_value = hyperbolic_val_loss_SUM/val_length

                distributional_train_loss_value = distributional_train_loss_SUM/train_length
                distributional_val_loss_value = distributional_val_loss_SUM/val_length

                distributional_train_metric_value = distributional_train_metric_SUM/train_length
                hyperbolic_train_metric_value = hyperbolic_train_metric_SUM/train_length

                distributional_val_metric_value = distributional_val_metric_SUM/val_length
                hyperbolic_val_metric_value = hyperbolic_val_metric_SUM/val_length

                classifier_loss_value = classifier_loss_sum/val_length

            else:
                train_loss_value = train_loss_SUM/train_weights_sum
                val_loss_value = val_loss_SUM/val_weights_sum

                hyperbolic_train_loss_value = hyperbolic_train_loss_SUM/train_weights_sum
                hyperbolic_val_loss_value = hyperbolic_val_loss_SUM/val_weights_sum

                distributional_train_loss_value = distributional_train_loss_SUM/train_weights_sum
                distributional_val_loss_value = distributional_val_loss_SUM/val_weights_sum

                distributional_train_metric_value = distributional_train_metric_SUM/train_weights_sum
                hyperbolic_train_metric_value = hyperbolic_train_metric_SUM/train_weights_sum

                distributional_val_metric_value = distributional_val_metric_SUM/val_weights_sum
                hyperbolic_val_metric_value = hyperbolic_val_metric_SUM/val_weights_sum
                

            self.checkpointManager(val_loss_value, epoch)

            if self.early_stopping:
                break

            losses_dict = {'Losses/Hyperbolic Losses': {'Train': hyperbolic_train_loss_value, 
                                                 'Val': hyperbolic_val_loss_value},
                           'Losses/Distributional Losses':  {'Train': distributional_train_loss_value,
                                                      'Val': distributional_val_loss_value},
                           'Losses/MTL-Losses': {'Train': train_loss_value,
                                          'Val': val_loss_value}
            }


            metric_dict = {'Metrics/Hyperbolic Metrics': {'Train': hyperbolic_train_metric_value, 
                                                 'Val': hyperbolic_val_metric_value},
                           'Metrics/Distributional Metrics':  {'Train': distributional_train_metric_value,
                                                                'Val': distributional_val_metric_value}
                            }
            self.log_losses(losses_dict = losses_dict, epoch = epoch + 1)

            self.log_losses(losses_dict=metric_dict, epoch = epoch + 1)

            print('{:^25}'.format('epoch {:^3}/{:^3}'.format(epoch, self.epochs)))
            print('{:^25}'.format('Train loss: {:.4f}, Val loss: {:.4f}, Min loss: {:.4f} at epoch: {}'.format(train_loss_value, 
                                                                                                    val_loss_value, 
                                                                                                    self.min_loss, 
                                                                                                    self.best_epoch)))
            print('{:^25}'.format('T_MHD: {:.4f}, V_MHD:{:.4f}'.format(hyperbolic_train_metric_value, hyperbolic_val_metric_value)))
            print('{:^25}'.format('T_MDD: {:.4f}, V_MDD:{:.4f}'.format(distributional_train_metric_value, distributional_val_metric_value)))
            print('classifier_loss: {:.4f}'.format(classifier_loss_value))


    def type_prediction_on_test(self, topn, test_data, entities, labels):

        checkpoint = torch.load(self.checkpoint_path)

        try:
            print('loading model checkpoint at epoch {}'.format(checkpoint['epoch']))
            with open(self.TSV_path, 'a') as out:
                out.write('-------------------------------------')
                out.write('best epoch: {}'.format(checkpoint['epoch']))
                out.write('-------------------------------------\n')


            with open(self.results_path, 'a') as out:
                out.write('-------------------------------------')
                out.write('best epoch: {}'.format(checkpoint['epoch']))
                out.write('-------------------------------------\n')
        except:
            pass

        self.load_state_dict(checkpoint['model_state_dict'])

        self.eval()

        self.topn = topn

        x = [[], []]
        class_preds = []
        i = 0
        print('... extract prediction on test ...')
        for i in tqdm(range(len(test_data['data']))):
            t = test_data['data'][i]
            pred, class_pred = self(t)
            x[0].extend(pred[0].detach().cpu().numpy())
            x[1].extend(pred[1].detach().cpu().numpy())
            class_preds.extend(class_pred.detach().cpu().numpy())

        self.emb = Embedding()            
        self.emb.set_embedding(self.datasetManager.concept_embeddings['hyperbolic'])
        self.emb.set_name('hyperbolic')
            
        self.compute_statistics(class_preds = class_preds)

        self.compute_classifier_prediction(class_preds, labels)

        self.compute_topn_predictions(class_preds, labels)
        
        super().compute_prediction(x, labels = labels, entities = entities)

    def compute_classifier_prediction(self, predictions, labels):
        
        tot = 0
        correct = 0

        preds = []
        for c in predictions:
            indexes = np.where(c > 0.5)
            class_preds = self.datasetManager.get_label_from_numeric(indexes)
            preds.extend(class_preds)

        measures, perfect_measures = self.compute_IR_metrics_classifier(predictions = preds, labels_lists = labels)

        self.save_IR_results(measures = measures, perfect = perfect_measures)

    def compute_IR_metrics_classifier(self, predictions, labels_lists, granularities = ['coarse', 'fine', 'ultrafine', 'all']):
           
        precision = {g: {t: 0 for t in self.topn} for g in granularities}
        recall = {g: {t: 0 for t in self.topn} for g in granularities}        
        
        precision_values = {g: {t: [] for t in self.topn} for g in granularities}
        recall_values = {g: {t: [] for t in self.topn} for g in granularities}
        
        perfect_score_precision = {g: {t: [] for t in self.topn} for g in granularities}
        perfect_score_recall = {g: {t: [] for t in self.topn} for g in granularities}
        perfect_score_fmeasure = {g: {t: [] for t in self.topn} for g in granularities}

        fmeasure = {g: {t: 0 for t in self.topn} for g in granularities}

        bar = tqdm(total=len(predictions))

        i = 0
        for pred, labels in zip(predictions, labels_lists):
            bar.set_description('getting IR results')
            for g in granularities:
                # neigh = self.emb.get_neigh(vector = pred, top = max(self.topn), include=g)
                
                if g != 'all':
                    filtered_labels = self.filter_labels(labels=labels, gran = g)
                else:
                    filtered_labels = labels
                if filtered_labels:
                    for t in self.topn:
                        new_neigh = pred
                        corrects = 0
                        for n in new_neigh:
                            if n in filtered_labels:
                                corrects += 1
                        if new_neigh:
                            prec = corrects / len(new_neigh)
                        else:
                            prec = 0

                        if len(filtered_labels) > t:
                            perfect_score_precision[g][t].append(1)
                            perfect_score_recall[g][t].append(t/len(filtered_labels))
                        else:
                            perfect_score_precision[g][t].append(len(filtered_labels)/t)
                            perfect_score_recall[g][t].append(1)

                        rec = corrects / len(filtered_labels)

                        precision_values[g][t].append(prec)
                        recall_values[g][t].append(rec)
                        
                        # print('pred: {}'.format(pred))
                        # print('labels: {}'.format(labels))
                        # print('filtered labels: {}'.format(filtered_labels))
                        # print('prec: {}'.format(prec))
                        # print('rec: {}'.format(rec))
                        # print('recalls: {}'.format(np.mean(recall_values[g][t])))
                        # print('------------------')
            bar.update(1)

            # if i > 9:
                # break

            i += 1
        
        bar.close()
        for g in granularities:
            for t in self.topn:
                precision[g][t] = np.mean(precision_values[g][t])
                recall[g][t] = np.mean(recall_values[g][t])
                if not recall[g][t] and recall[g][t] != 0:
                    recall[g][t] = 0

                fmeasure[g][t] = self.f1(p=precision[g][t], r=recall[g][t])

                perfect_score_precision[g][t] = np.mean(perfect_score_precision[g][t])
                perfect_score_recall[g][t] = np.mean(perfect_score_recall[g][t])
                perfect_score_fmeasure[g][t] = self.f1(p=perfect_score_precision[g][t], r=perfect_score_recall[g][t])

        print('recall: {}'.format(recall))

        return [precision, recall, fmeasure], [perfect_score_precision, perfect_score_recall, perfect_score_fmeasure]

    def compute_topn_predictions(self, class_preds, labels):

        bar = tqdm(total=len(self.topn) * len(labels))
        bar.set_description('compute topn for classifier')

        precision, recall, fmeasure = {'all': {t: [] for t in self.topn}}, {'all': {t: [] for t in self.topn}}, {'all': {t: [] for t in self.topn}}

        perfect_precision, perfect_recall, perfect_fmeasure = {'all': {t: [] for t in self.topn}}, {'all': {t: [] for t in self.topn}}, {'all': {t: [] for t in self.topn}}
        
        for t in self.topn:
            for c, l in zip(class_preds, labels):
                pred = self.extract_top(c, t)
                
                m, p = self.compute_measures_and_perfect(prediction=pred, true_labels=[l])

                precision['all'][t].append(m['precision'][0])
                recall['all'][t].append(m['recall'][0])
                fmeasure['all'][t].append(m['fmeasure'][0])

                perfect_precision['all'][t].append(p['precision'][0])
                perfect_recall['all'][t].append(p['recall'][0])
                perfect_fmeasure['all'][t].append(p['fmeasure'][0])
                
                bar.update(1)
        
            precision['all'][t] = np.mean(precision['all'][t])
            recall['all'][t] = np.mean(recall['all'][t])
            fmeasure['all'][t] = np.mean(fmeasure['all'][t])


            perfect_precision['all'][t] = np.mean(perfect_precision['all'][t])
            perfect_recall['all'][t] = np.mean(perfect_recall['all'][t])
            perfect_fmeasure['all'][t] = np.mean(perfect_fmeasure['all'][t])

        measures = [precision, recall, fmeasure]
        perfect_measures = [perfect_precision, perfect_recall, perfect_fmeasure]

        self.save_IR_results(measures = measures, perfect = perfect_measures)

        bar.close()

    def extract_top(self, class_preds, t):
        
        indexes = class_preds.argsort()[-t:][::-1]

        return self.datasetManager.get_label_from_numeric([indexes])

    
    def compute_statistics(self, class_preds):
        for t in self.topn:
            value_sum = 0
            for p in class_preds:
                value = sorted(p, reverse=True)[t - 1]
                value_sum += value

            self.write_statistics(t, value_sum/len(class_preds))

        lens_sum = 0
        for p in class_preds:
            indexes = np.where(p > 0.5)
            c_p = self.datasetManager.get_label_from_numeric(indexes)
            lens_sum += len(c_p[0])

        average_preds = lens_sum/len(class_preds)

        with open(self.results_path, 'a') as out:
            out.write('\n average prediction with threshold setted as 0.5 : {}\n'.format(average_preds))


    def write_statistics(self, top, value):
        with open(self.results_path, 'a') as out:
            out.write('\n average inferior bound to produce {} predictions: {}\n'.format(top, value))

        # with open(self.TSV_path, 'a') as out:
            # out.write('{}\t'.format(value))
        

