import numpy as np
import copy
import math

class NeuralNetwork():
    def __init__(self):
        self.layers = []
        self.parameters = {}
        self.cache = []
        self.grads = {}
        self.params = {}
        self.mini_batches = []
    
    def add(self,units,activation):
        self.layers.append([units,activation])

    def init_weights(self,x_dim):
        np.random.seed(1)
        self.layers.insert(0,[x_dim,'relu'])
        
        L = len(self.layers) -1 

        for l in range(1, L + 1):
            self.parameters['W' + str(l)] = np.random.randn(self.layers[l][0],self.layers[l-1][0]) * np.sqrt(2/self.layers[l-1][0])
            self.parameters['b' + str(l)] = np.zeros((self.layers[l][0],1))



    def relu(self,x):
        return np.maximum(0,x)

    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))

    def relu_backward(self,dA,Z):
        dZ = np.array(dA,copy=True)
        dZ [Z <= 0 ] = 0
        return dZ
    
    def sigmoid_backward(self,dA,Z):
        x = 1 / (1+np.exp(-Z))
        return dA * x *(1-x)

    def forward_propagation(self,X):
        self.cache = []
        L = len(self.parameters) // 2
        
        A = X

        for l in range(1,L):
            a_prev = A
            W = self.parameters[f"W{l}"]
            b = self.parameters[f"b{l}"]
            Z = np.dot(W,a_prev) + b
            
            if self.layers[l][1] == 'relu':
                A = self.relu(Z)
            elif self.layers[l][1] == 'sigmoid':
                A = self.sigmoid(Z)
                
            self.cache.append((a_prev,W,b,Z))

        W = self.parameters[f"W{L}"]
        b = self.parameters[f"b{L}"]
        Z = np.dot(W,A) + b

        if self.layers[L][1] == 'relu':
            A = self.relu(Z)
        elif self.layers[L][1] == 'sigmoid':
            A = self.sigmoid(Z)
            
        self.cache.append((A,W,b,Z))
        return A


    def loss(self,Output,Y):
        m = Y.shape[1]
        if self.layers[-1][1] == 'relu':
            cost = 1/m * np.sum(np.abs(Y-Output))
            cost = np.squeeze(cost)
        elif self.layers[-1][1] == 'sigmoid':
            epsilon = 1e-8
            cost = -1/m * np.sum(Y*np.log(Output+epsilon) + (1-Y) * np.log((1-Output)+epsilon))
            cost = np.squeeze(cost)
        return cost


    def compute_grads(self,dZ,A,W):
        m = A.shape[1]
        dA = np.dot(W.T,dZ)
        dW = 1/m * np.dot(dZ,A.T)
        db = 1/m * np.sum(dZ,axis=1,keepdims=True)
        return dA,dW,db

    def backward_propagation(self,Y,A):
        L = len(self.cache)

        Y = Y.reshape(A.shape)
        epsilon = 1e-8
        #dA = - (np.divide(Y,A) - np.divide(1-Y,1-A))
        dA = - (np.divide(Y, A + epsilon) - np.divide(1 - Y, 1 - A + epsilon))

        A,W,b,Z = self.cache[L-1] 
        if self.layers[L-1][1] == 'relu':
            dZ = self.relu_backward(dA,Z)
        elif self.layers[L-1][1] == 'sigmoid':
            dZ = self.sigmoid_backward(dA,Z)

        dA,dW,db = self.compute_grads(dZ,A,W)
        self.grads['dW' + str(L)] = dW
        self.grads['db' + str(L)] = db
        self.grads['dA' + str(L-1)] = dA

        for l in reversed(range(L-1)):
            A,W,b,Z = self.cache[l]
            if self.layers[l][1] == 'relu':
                dZ = self.relu_backward(dA,Z)
            elif self.layers[l][1] == 'sigmoid':
                dZ = self.sigmoid_backward(dA,Z)
    
            dA,dW,db = self.compute_grads(dZ,A,W)
            self.grads['dW' + str(l+1)] = dW
            self.grads['db' + str(l+1)] = db
            self.grads['dA' + str(l)] = dA


    def update_parameters_gd(self,lr):
        self.params = copy.deepcopy(self.parameters)
        L = len(self.parameters) // 2
    
        for l in range(1,L+1):
            self.params[f"W{l}"] = self.params[f"W{l}"] - lr * self.grads[f"dW{l}"]
            self.params[f"b{l}"] = self.params[f"b{l}"] - lr * self.grads[f"db{l}"]

        self.parameters = self.params


    def lr_decay(self,lr,epoch,decay_rate=1,time_interval=1000):
        learning_rate = (1 / (1+decay_rate * np.floor(epoch/time_interval))) * lr
        return learning_rate

    def update_parameters_adam(self,lr,t,beta1=0.9,beta2=0.999,epsilon=1e-8):
        self.params = copy.deepcopy(self.parameters)
        L = len(self.parameters) // 2

        v_corrected = {}
        s_corrected = {}
        
        for l in range(1,L+1):
            self.v['dW'+str(l)] = beta1 * self.v['dW'+str(l)] + (1-beta1) * self.grads['dW'+str(l)]
            self.v['db'+str(l)] = beta1 * self.v['db'+str(l)] + (1-beta1) * self.grads['db'+str(l)]
            v_corrected['dW'+str(l)] = self.v['dW'+str(l)] / (1-beta1**t)
            v_corrected['db'+str(l)] = self.v['db'+str(l)] / (1-beta1**t)
            self.s['dW'+str(l)] = beta2 * self.s['dW'+str(l)] + (1-beta2) * self.grads['dW'+str(l)]**2
            self.s['db'+str(l)] = beta2 * self.s['db'+str(l)] + (1-beta2) * self.grads['db'+str(l)]**2
            s_corrected['dW'+str(l)] = self.s['dW'+str(l)] / (1-beta2**t)
            s_corrected['db'+str(l)] = self.s['db'+str(l)] / (1-beta2**t)
            self.params[f"W{l}"] = self.params[f"W{l}"] - lr * (v_corrected['dW'+str(l)]) / (np.sqrt(s_corrected['dW'+str(l)]) + epsilon) 
            self.params[f"b{l}"] = self.params[f"b{l}"] - lr * (v_corrected['db'+str(l)]) / (np.sqrt(s_corrected['db'+str(l)]) + epsilon) 

        self.parameters = self.params

    def predict(self,X):
        A = self.forward_propagation(X)
        if self.layers[-1][1] == 'relu':
            return A
        elif self.layers[-1][1] == 'sigmoid':
            pred = (A > 0.5).astype(int)
            return pred


    def compile(self,X):
        x_dim = X.shape[0]
        self.init_weights(x_dim)
        self.initialize_adam()

    def make_batches(self,batch_size,X,Y):
        m = X.shape[1]
        self.mini_batches = []
        permutation = list(np.random.permutation(m))
        shuffled_X = X[:, permutation]
        shuffled_Y = Y[:, permutation].reshape((1, m))

        n_batches = math.floor( m / batch_size)

        for k in range(0,n_batches):
            x_temp = shuffled_X[:,k*batch_size:(k+1)*batch_size]
            y_temp = shuffled_Y[:,k*batch_size:(k+1)*batch_size]

            temp = (x_temp,y_temp)
            self.mini_batches.append(temp)


        if m % batch_size != 0:
            x_temp = shuffled_X[:,(k+1)*batch_size:]
            y_temp = shuffled_Y[:,(k+1)*batch_size:]

            temp = (x_temp,y_temp)
            self.mini_batches.append(temp)

    
    def fit(self,X,Y,epochs=20,learning_rate=0.0001,batch_size=32,optimizer='adam'):
        self.cache = []
        self.grads = {}
        self.params = {}
        self.make_batches(batch_size,X,Y)

        t = 0
        
        history = {}
        history['loss'] = []
        
        for i in range(epochs):
            for batch in self.mini_batches:
                (batch_X,batch_Y) = batch
                A = self.forward_propagation(batch_X)
                loss = self.loss(A,batch_Y)

                self.backward_propagation(batch_Y,A)

                if optimizer == 'gd':
                    self.update_parameters_gd(learning_rate)
                else:
                    t = t + 1
                    self.update_parameters_adam(learning_rate,t=t)


            learning_rate = self.lr_decay(learning_rate,i)
            
            if i %10 ==0:
                print(f"Epochs : {i} , loss = {loss:.5f}")
                history['loss'].append(loss)
                
        return history


    def initialize_adam(self):
        self.v = {}
        self.s = {}
        L = len(self.parameters) // 2
        
        
        for l in range(1,L+1):
            self.v['dW'+str(l)] = np.zeros(self.parameters['W'+str(l)].shape)
            self.v['db'+str(l)] = np.zeros(self.parameters['b'+str(l)].shape)
            self.s['dW'+str(l)] = np.zeros(self.parameters['W'+str(l)].shape)
            self.s['db'+str(l)] = np.zeros(self.parameters['b'+str(l)].shape)
            
