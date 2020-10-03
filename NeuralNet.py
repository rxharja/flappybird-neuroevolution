import numpy as np


class NeuralNet:
    '''
        NeuroEvolution neural net implementing 4 inputs,
        a variable number of hidden layers and nodes,
        and one output.
    '''
    def __init__(self, hidden_layers: int, hidden_nodes: int, input: np.array, bias=1):
        self.hidden_layers = hidden_layers
        self.hidden_nodes = hidden_nodes
        self.X = np.array(input)
        self.bias = bias
        self.theta = None
        self.final_bias = None
        self.initialize_weights()
        
    def initialize_weights(self):
        rows = 5 + ((self.hidden_layers - 1) * self.hidden_nodes)
        self.theta = 2 * np.random.random_sample((rows + (self.hidden_layers), self.hidden_nodes)) - 1
        self.final_bias = 2 * np.random.random_sample() - 1
        self.weights_init = True

    def sigmoid(self, z: np.array):
        return np.array(1/(1 + np.exp(-z)), dtype=np.float128)
    
    def forward_prop(self):            
        a = None; sub_a = None; sub_z = None; sub_theta = None
        a = np.zeros(((self.hidden_layers * self.hidden_nodes) + 4, 1))
        a[0:4] = np.reshape(self.X, (4,1))
        
        a1 = np.c_[self.bias, a[0:4].T].T
        theta1 = self.theta[0:5,:]
        z2 = theta1.T @ a1
        a2 = self.sigmoid(z2)

        sub_a = a2
        row_start = 5
        for i in range(0, self.hidden_layers - 1):
            sub_a = np.c_[self.bias, sub_a.T].T
            sub_theta = self.theta[row_start:row_start+self.hidden_nodes+1,:]
            sub_z = sub_theta.T @ sub_a
            sub_a = self.sigmoid(sub_z)
            row_start += self.hidden_nodes + 1
        
        thetaL = np.c_[self.final_bias, self.theta[row_start:,:]]
        sub_a = np.c_[self.bias, sub_a.T].T
        zL = thetaL @ sub_a
        return self.sigmoid(zL)