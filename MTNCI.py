from torch import nn
import geoopt
from geooptModules import MobiusLinear

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
    def __init__(self, hidden_dim, dims, mode):
        super().__init__()
        self.out = nn.ModuleList()
        
        self.dropout = nn.Dropout(p=0.1).cuda()
        self.leaky_relu = nn.ReLU().cuda()
        self.bns = nn.ModuleList()
        
        prec = hidden_dim
        
        for dim in dims:
            if mode == 'hyper':
                self.out.append(MobiusLinear(prec, dim).cuda())
            elif mode == 'euclid':
                self.out.append(nn.Linear(prec, dim).cuda())
            else:
                print('ERROR: NO MODE SELECTED')
                
            self.bns.append(nn.BatchNorm1d(dim).cuda())
            prec = dim
            
    def forward(self, x):
#         x = x.double()
        for i in range(len(self.out) - 1):
            x = self.leaky_relu(self.out[i](x))
        out = self.out[-1](x)
        return out

class MTNCI(nn.Module):    
    def __init__(self,
                 input_d, 
                 dims = None,
                 dis_dims = 10,
                 hyp_dims = 2,
                 dropout_prob = 0.2,):
        
        super().__init__()
        
        self.common_network = CommonLayer(input_d=input_d,
                                          dims = dims,
                                          dropout_prob=dropout_prob)

        self.dis_out_layer = RegressionOutput(hidden_dim=dims[-1],
                                              dims=dis_dims,
                                              mode = 'euclid')
       
        self.hyp_out_layer = RegressionOutput(hidden_dim=dims[-1],
                                              dims=hyp_dims,
                                              mode = 'hyper')
            
    def forward(self, x):
        x = self.common_network(x)
        
        outDis = self.dis_out_layer(x)
        outHyp = self.hyp_out_layer(x)
        
        return outDis, outHyp    

    def voting(self, vectors, hyperbolic_embedding, distributional_embedding):
        