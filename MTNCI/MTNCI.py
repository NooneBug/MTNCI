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
    
    def set_hyperparameters(self, epochs, times = 1, perc = 1, patience = 50, weighted = False, regularized = False):
        self.epochs = epochs
        self.weighted = weighted
        self.regularized = regularized
        self.patience = patience
        self.early_stopping = False

        self.PERC = perc
        self.TIMES = times

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
        distributional_val_metric_manager = self.get_prediction_manager(loss_name=self.metric_name_dict['hyperbolic'])

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
                        hyperbolic_train_loss_SUM += torch.sum(hyperbolic_train_loss * (self.llambdas['hyperbolic'])).item()
                        
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
                                                                'Val': distributional_val_metric_value},
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
        entities_dict = {e: [v for v, entity in zip(predictions, entities) if entity == e] for e in set(entities)}
        entities_labels = {e: l for e, l in zip(entities, labels)}

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
            out = open(self.TSV_path, 'w')
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
                       'normalized_hyperbolic_distance': 'NHYPD',
                       'regularized_hyperbolic_distance': 'RHYPD',
                       'hyperboloid_distance' : 'LORENTZD',
                       'multilabel_Minimum_Poincare': 'HMML',
                       'multilabel_Minimum_cosine': 'DMML'
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
        elif loss_name == self.LOSSES['normalized_hyperbolic_distance']:
            self.selected_loss = normalizedPoincareDistanceLoss(device = self.device)
        elif loss_name == self.LOSSES['regularized_hyperbolic_distance']:
            self.selected_loss = regularizedPoincareDistanceLoss(device = self.device)
        elif loss_name == self.LOSSES['multilabel_Minimum_Poincare']:
            self.selected_loss = multilabelMinimumPoincareDistanceLoss(device=self.device)
        elif loss_name == self.LOSSES['multilabel_Minimum_cosine']:
            self.selected_loss = multilabelMinimumCosineDistanceLoss(device=self.device)
        
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

class multilabelMinimumPoincareDistanceLoss(normalizedPoincareDistanceLoss):
    def compute_loss(self, true, pred):
        # RAGIONARE PER BATCH
        for i, t in enumerate(true):
            loss = super().compute_loss(true = t, pred = pred)
            
            if i == 0:
                min_loss = loss
            elif loss < min_loss:
                min_loss = loss

        return min_loss

class multilabelMinimumCosineDistanceLoss(self, true, pred):


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
        contexts, positions, context_len = input[0], input[1].double(), input[2]
        mentions, mention_chars = input[3], input[4]
        type_indexes = input[5]
                
        mention_vec = self.mention_encoder(mentions, mention_chars, self.word_lut)
        
        context_vec, attn = self.context_encoder(contexts, positions, context_len, self.word_lut)

        input_vec = torch.cat((mention_vec, context_vec), dim=1)
        
        return super().forward(input_vec)
    
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
    pass
