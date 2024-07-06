import numpy as np

class NeuralNetwork():
    def __init__(self):
        self.layers = []
        self.parameters = {}
        self.cache = []
        self.grads = {}
        self.params = {}
    
    def add(self,units,activation):
        self.layers.append([units,activation])

    def init_weights(self,x_dim):

        self.layers.insert(0,[x_dim,'None'])
        
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
            cost = 1/m * np.sum((Y-Output)**2)
        elif self.layers[-1][1] == 'sigmoid':
            cost = -1/m * np.sum(Y*np.log(Output) + (1-Y) * np.log(1-Output))
            cost = np.squeeze(cost)
        return cost


    def compute_grads(dZ,A,W):
        m = A.shape[1]
        dA = np.dot(W.T,dZ)
        dW = 1/m * np.dot(dZ,A.T)
        db = 1/m * np.sum(dZ,axis=1,keepdims=True)
        return dA,dW,db

    def backward_propagation(self,Y,A):

        L = len(self.cache)

        Y = Y.reshape(A.shape)
        dA = - (np.divide(Y,A) - np.divide(1-Y,1-A))

        A,W,b,Z = self.cache[L-1] 
        if self.layers[L][1] == 'relu':
            dZ = self.relu_backward(dA,Z)
        elif self.layers[L][1] == 'sigmoid':
            dZ = self.sigmoid_backward(dA,Z)

        dA,dW,dB = self.compute_grads(dZ,A,W)
        self.grads['dW' + str(L)] = dW
        self.grads['db' + str(L)] = db
        self.grads['dA' + str(L-1)] = dA

        for l in reversed(range(L-1)):
            A,W,b,Z = self.cache[l]
            if self.layers[l+1][1] == 'relu':
                dZ = self.relu_backward(dA,Z)
            elif self.layers[l+1][1] == 'sigmoid':
                dZ = self.sigmoid_backward(dA,Z)
    
            dA,dW,dB = self.compute_grads(dZ,A,W)
            self.grads['dW' + str(l+1)] = dW
            self.grads['db' + str(l+1)] = db
            self.grads['dA' + str(l)] = dA


    def update_parameters(self,lr):
        self.params = copy.deepcopy(self.parameters)
        L = len(parameters) // 2
    
        for l in range(L):
            self.params[f"W{l+1}"] = self.params[f"W{l+1}"] - lr * self.grads[f"dW{l+1}"]
            self.params[f"b{l+1}"] = self.params[f"b{l+1}"] - lr * self.grads[f"db{l+1}"]


    def predict(self,X):
        A = self.forward_propagation(X)
        if self.layers[-1][1] == 'relu':
            return A
        elif self.layers[-1][1] == 'sigmoid':
            pred = (A > 0.5).astype(int)
            return pred


    def fit(self,X,Y,epochs=1000,learning_rate=0.001):
        x_dim = X.shape[0]
        self.init_weights(x_dim)

        for i in range(epochs):
            A = self.forward_propagation(X)
            loss = self.loss(A,Y)
            self.backward_propagation(Y,A)
            self.update_parameters(learning_rate)

            if i %100 ==0:
                print(f"Epochs : {i} , loss = {loss}")
