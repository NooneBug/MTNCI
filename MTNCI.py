from torch import nn
import geoopt
from geooptModules import MobiusLinear
from torch.utils.data import Dataset, DataLoader
from geoopt.optim import RiemannianAdam


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
                dropout_prob = 0.2):
        
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


    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_lambda(self, llambdas):
        self.llambdas = llambdas

    def get_multitask_loss(self, prediction_dict):
        loss = 0
        for k in prediction_dict.keys():
            loss += prediction_dict[k] * self.llambda[k]
        
        return torch.sum(loss)
    
    def set_hyperparameters(self, epochs, weighted = False):
        self.epochs = epochs
        self.weighted = weighted


    def train(self):

        distributional_prediction_manager = Prediction()
        distributional_prediction_manager.select_loss(distributional_prediction_manager.LOSSES['cosine_dissimilarity'])

        hyperbolic_prediction_manager = Prediction()
        hyperbolic_prediction_manager.select_loss(distributional_prediction_manager.LOSSES['hyperbolic_distance'])

        for epoch in range(epochs):
            train_it = iter(self.trainloader)
            val_it = iter(self.valloader)
            
            train_loss_SUM = 0
            val_loss_SUM = 0
            
            distributional_train_loss_SUM = 0
            distributional_val_loss_SUM = 0
            
            hyperbolic_train_loss_SUM = 0
            hyperbolic_val_loss_SUM = 0
            
            for batch_iteration in range(len(self.trainloader)):
                x, labels, targets = next(train_it)
                
                ######################
                ####### TRAIN ########
                ######################
                
                
                self.optimizer.zero_grad()
                
                self.train()
                
                output = self(x)

                distributional_prediction_manager.set_prediction(predictions = output[0],
                                                                 true_values = targets['distributional'])

                hyperbolic_prediction_manager.set_prediction(predictions = output[1], 
                                                             true_values = targets['hyperbolic'])                


                distributional_train_loss = distributional_prediction_manager.compute_loss()
                
                hyperbolic_train_loss = hyperbolic_prediction_manager.compute_loss()

                distributional_train_loss_SUM += torch.sum(distributional_train_loss * self.llambdas['distributional']).item()
                hyperbolic_train_loss_SUM += torch.sum(hyperbolic_train_loss * (1 - self.llambdas['hyperbolic'])).item()
                
                train_loss = self.get_multitask_loss({'distributional': distributional_train_loss, 
                                                      'hyperbolic': hyperbolic_train_loss})
                
                train_loss_SUM += train_loss.item()
                train_loss.backward()
                optimizer.step()
                
                
            else:

                ######################
                ######## VAL #########
                ######################
                
                with torch.no_grad():
                    model.eval()   
                    
                    for batch_iteration in range(len(valloader)):
                        x, labels, targets = next(val_it)


                        output = self(x)

                        distributional_prediction_manager.set_prediction(predictions = output[0],
                                                                        true_values = targets['distributional'])

                        hyperbolic_prediction_manager.set_prediction(predictions = output[1], 
                                                                    true_values = targets['hyperbolic']) 
                        
                        distributional_val_loss = distributional_prediction_manager.compute_loss()
                        
                        hyperbolic_val_loss = hyperbolic_prediction_manager.compute_loss()
                        
                        distributional_val_loss_SUM += torch.sum(distributional_val_loss * self.llambdas['distributional']).item()
                        hyperbolic_val_loss_SUM += torch.sum(hyperbolic_val_loss * (1 - self.llambdas['hyperbolic'])).item()
                        
                        val_loss = self.get_multitask_loss({'distributional': distributional_val_loss, 
                                                            'hyperbolic': hyperbolic_val_loss})
                        
                        val_loss_SUM += val_loss.item()
                
            print('{:^15}'.format('epoch {:^3}/{:^3}'.format(epoch, epochs)))
            print('{:^15}'.format('Train loss: {:.4f}, Val loss: {:.4f}'.format(train_loss_SUM/len(c.X_train), 
                                                                                val_loss_SUM/len(c.X_val)
                                                                            )
                                )
                )
      


class Prediction:

    def __init__(self):
        self.LOSSES = {'cosine_dissimilarity': 'COSD',
                       'hyperbolic_distance': 'HYPD',
                       'regularized_hyperbolic_distance': 'RHYPD'
        
        }

    def set_prediction(self, predictions, true_values):
        self.predictions = predictions
        self.true_values = true_values
    
    def select_loss(self, loss_name):
        if loss_name == self.LOSSES['cosine_dissimilarity']:
            self.selected_loss = cosineLoss()
        elif loss_name == self.LOSSES['hyperbolic_distance']:
            self.select_loss = poincareDistanceLoss()
        elif loss_name == self.LOSSES['regularized_hyperbolic_distance']:
            self.select_loss = regularizedPoincareDistanceLoss()

    def compute_loss(self):
        self.select_loss.compute_loss(true = self.true_values, 
                                      pred = self.predictions)


class Loss():
    def compute_loss(self):


class cosineLoss(Loss):
    def compute_loss(self):
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
        acos = acosh(1  + frac)
        
        return acos

    def acosh(x):
        return torch.log(x + torch.sqrt(x**2-1))

    def mse(y_pred, y_true, regul):    
        mse_loss = nn.MSELoss()
        return mse_loss(y_pred, y_true)

class regularizedPoincareDistanceLoss(poincareDistanceLoss):

    def set_regularization(self, regul):
        self.regul = regul

    def compute_loss(self, true, pred):
        acosh = super().compute_loss(true = true, pred = pred)

        l0 = torch.tensor(1., device = device)
        l1 = torch.tensor(1., device = device)
        
        if sum(regul) > 1:
            
            true_perm = y_true[torch.randperm(y_true.size()[0])]
            
            l0 = torch.abs(hyperbolic_loss(y_pred, true_perm, regul=[0, 0, 1])[0] - hyperbolic_loss(y_true, true_perm, regul = [0, 0, 1])[0])
            l1 = mse(y_pred, y_true, 0)
        
        return acos**self.regul[2] + l0 * self.regul[0] + l1 * self.regul[1]


class MobiusLinear(torch.nn.Linear):
    def __init__(self, *args, nonlin=None, ball=None, c=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        # for manifolds that have parameters like Poincare Ball
        # we have to attach them to the closure Module.
        # It is hard to implement device allocation for manifolds in other case.
        self.ball = create_ball(ball, c)
        if self.bias is not None:
            self.bias = geoopt.ManifoldParameter(self.bias, manifold=self.ball)
        self.nonlin = nonlin
        self.reset_parameters()

    def forward(self, input):
        return mobius_linear(
            input,
            weight=self.weight,
            bias=self.bias,
            nonlin=self.nonlin,
            ball=self.ball,
        )

    @torch.no_grad()
    def reset_parameters(self):
        torch.nn.init.eye_(self.weight)
        self.weight.add_(torch.rand_like(self.weight).mul_(1e-3))
        if self.bias is not None:
            self.bias.zero_()


# package.nn.functional.py
def mobius_linear(input, weight, bias=None, nonlin=None, *, ball: geoopt.PoincareBall):
    output = ball.mobius_matvec(weight, input)
    if bias is not None:
        output = ball.mobius_add(output, bias)
    if nonlin is not None:
        output = ball.logmap0(output)
        output = nonlin(output)
        output = ball.expmap0(output)
    return output

def create_ball(ball=None, c=None):
    """
    Helper to create a PoincareBall.
    Sometimes you may want to share a manifold across layers, e.g. you are using scaled PoincareBall.
    In this case you will require same curvature parameters for different layers or end up with nans.
    Parameters
    ----------
    ball : geoopt.PoincareBall
    c : float
    Returns
    -------
    geoopt.PoincareBall
    """
    if ball is None:
        assert c is not None, "curvature of the ball should be explicitly specified"
        ball = geoopt.PoincareBall(c)
    # else trust input
    return ball