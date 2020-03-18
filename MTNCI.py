from torch import nn
import torch
import geoopt
from geooptModules import MobiusLinear, mobius_linear, create_ball
from torch.utils.data import Dataset, DataLoader
from geoopt.optim import RiemannianAdam
from tensorBoardManager import TensorBoardManager
from abc import ABC, abstractmethod

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

    def train_model(self):

        train_length = len(self.datasetManager.X_train)
        val_length = len(self.datasetManager.X_val)

        
        distributional_prediction_train_manager = Prediction(device = self.device)
        distributional_prediction_train_manager.select_loss(distributional_prediction_train_manager.LOSSES['cosine_dissimilarity'])

        hyperbolic_prediction_train_manager = Prediction(device = self.device)
        hyperbolic_prediction_train_manager.select_loss(distributional_prediction_train_manager.LOSSES['hyperbolic_distance'])

        distributional_prediction_val_manager = Prediction(device = self.device)
        distributional_prediction_val_manager.select_loss(distributional_prediction_val_manager.LOSSES['cosine_dissimilarity'])

        hyperbolic_prediction_val_manager = Prediction(device = self.device)
        hyperbolic_prediction_val_manager.select_loss(distributional_prediction_val_manager.LOSSES['hyperbolic_distance'])

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
            
            train_loss_value = train_loss_SUM/train_length
            val_loss_value = val_loss_SUM/val_length

            hyperbolic_train_loss_value = hyperbolic_train_loss_SUM/train_length
            hyperbolic_val_loss_value = hyperbolic_val_loss_SUM/val_length

            distributional_train_loss_value = distributional_train_loss_SUM/train_length
            distributional_val_loss_value = distributional_val_loss_SUM/val_length


            losses_dict = {'Losses/Hyperbolic Losses': {'Train': hyperbolic_train_loss_value, 
                                                 'Val': hyperbolic_val_loss_value},
                           'Losses/Distributional Losses':  {'Train': distributional_train_loss_value,
                                                      'Val': distributional_val_loss_value},
                           'Losses/MTL-Losses': {'Train': train_loss_value,
                                          'Val': val_loss_value}
            }


            self.log_losses(losses_dict = losses_dict, epoch = epoch + 1)

            print('{:^15}'.format('epoch {:^3}/{:^3}'.format(epoch, self.epochs)))
            print('{:^15}'.format('Train loss: {:.4f}, Val loss: {:.4f}'.format(train_loss_SUM/len(self.datasetManager.X_train), 
                                                                                val_loss_SUM/len(self.datasetManager.X_val)
                                                                            )
                                )
                )
    def log_losses(self, losses_dict, epoch):
        for k, subdict in losses_dict.items():
            list_of_losses = [subdict['Train'], subdict['Val']]

            self.tensorBoardManager.log_losses(main_label= k, 
                                                list_of_losses = list_of_losses,
                                                list_of_names = ['Train', 'Val'], 
                                                epoch = epoch)

    

class Prediction:

    def __init__(self, device, weighted = False):
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

# class lorentzDistanceLoss(Loss):

#     def acosh(self, x):
#         return torch.log(x + torch.sqrt(x**2-1))

#     def hyperboloid_projection(v, r = 1):
#         n = norm(v)
#         t = [(r**2 + (n ** 2)) / (r**2 - (n ** 2))]
#         projected = [(2 * r**2 * vs) /(r**2 - (n ** 2)) for vs in v]
#         projected.extend(t)
#         return np.array(projected)

#     def inverse_projection(v, r):
#         return np.array([vs/(r**2 + v[-1]) for vs in v[:-1]])

#     def compute_loss(self, pred, true):
#         hyperboloid_pred = self.hyperboloid_projection(v = pred)
        