import numpy as np
import math #mathematics
import matplotlib.pyplot as plt #GRAPHIKI

#
class Layer:
    def __init__(self, input_num, neuron_num, o_fun, o_fun_point):
        self.input_num = input_num
        self.neuron_num = neuron_num
        self.weights = np.random.rand(input_num, neuron_num)
        self.s = np.array(np.zeros([neuron_num]), ndmin=2)
        self.o = np.array(np.zeros([neuron_num]), ndmin=2)
        self.bra_delta = np.array(np.zeros([neuron_num]),ndmin=2)
        self.activation_fun = o_fun
        self.activation_fun_point = o_fun_point #proizvodnaya ot f activazii
        pass
    def weighted_sum(self, in_x):
        self.s = in_x.dot(self.weights)
        pass
    def activate_s(self):
        self.o = self.activation_fun(self.s)
        pass
    def activate_s_point(self):
        return self.activation_fun_point(self.o)
    
    def calculate_out(self, in_x):
        self.weighted_sum(in_x)
        self.activate_s()
        pass
############

#
class Input_layer(Layer):
    def __init__(self, input_num):
        #
        def inp_layer_f(x):
            return x
        #
        def inp_layer_f_point(x):
            return 0
        #
        super().__init__(input_num, input_num, inp_layer_f, inp_layer_f_point)
        self.weights = np.eye(input_num)        
#
class NN:
    #init takes array of neuron nums d[0] = num_of_in_x, d[1] = neuron num of first hidden Layer...!!!
    #razmernost f menee d na 1 i soderzhit f[0] = activ fun nulevogo slouy
    def __init__(self, d, f, f_p):
        self.d = d
        self.layer = [Input_layer(d[0])]
        #making of input layer
        for i in range(len(d)-1):
            self.layer.append(Layer(d[i],d[i+1],f[i], f_p[i]))
            
        pass
    def get_y(self):
        return self.layer[len(self.d)-1].o
        pass
    def get_w(self):
        n = 0
        for i in self.layer:
            print('w[' , n, ']', ' = ', i.weights)
            n+=1
        pass
    def make_calculation(self, bra_x):
        in_x = np.array(bra_x, ndmin = 2)
        for i in self.layer:
            i.calculate_out(in_x)
            in_x = i.o
        return in_x
        pass
    def learn(self, lr, epochs, x_arr, target_bras_arr):
        for e in range(epochs):
            for i in range(x_arr.shape[0]):
                bra_x = np.array(x_arr[i], ndmin = 2)
                E = target_bras_arr[i,:] - self.make_calculation(bra_x)
                m = len(self.layer)
                self.layer[m-1].bra_delta = np.array(E*self.layer[m-1].activate_s_point(),ndmin =2)
                self.layer[m-1].weights += lr*self.layer[m-2].o.transpose().dot(self.layer[m-1].bra_delta)
                for j in range(m-2):
                    ind = m-j-2
                    self.layer[ind].bra_delta = self.layer[ind].activate_s_point()*self.layer[ind+1].bra_delta.dot(self.layer[ind+1].weights.transpose())
                    self.layer[ind].weights += lr*self.layer[ind-1].o.transpose().dot(self.layer[ind].bra_delta)
#                     if ind==0:
#                         self.layer[ind].bra_delta = self.layer[ind].activate_s_point()*self.layer[ind+1].bra_delta.dot(self.layer[ind+1].weights.transpose())
#                         self.layer[ind].weights += lr*bra_x.transpose().dot(self.layer[ind].bra_delta)
#                     else:
#                         self.layer[ind].bra_delta = self.layer[ind].activate_s_point()*self.layer[ind+1].bra_delta.dot(self.layer[ind+1].weights.transpose())
#                         self.layer[ind].weights += lr*self.layer[ind-1].o.transpose().dot(self.layer[ind].bra_delta)
                    pass
##################################################################################
                ##################################################################
                ##################################################################
                #USING EXAMPLE
                #WITH XOR                  
                
    
# Activation functions and their points (proizvodnie):
def sigmoidal(x):
    return 1/(1+np.exp(-x))
def sigmoidal_point(o):
    return o*(1-o)
def heaviside(x):
    bias = 3
    if x<bias:
        return 0
    else:
        return 1
    pass
def heaviside_point(o):
    return 1

#MAIN MAIN MAIN
#
# opening file with train_data
data_file = open("data.csv",'r')
str_list_data = data_file.readlines()
data_file.close()

#array of train data
data_arr = np.zeros([4,3])
for i in range(len(str_list_data)):
  data_arr[i] = str_list_data[i].split(',')

# x_arr is bra-input-vectors, one under one
x_arr = np.array(data_arr[:, 0:2])
#bra_y is bra-target-vectors, one under one
bra_y = np.array(data_arr[:, 2], ndmin = 2).T
epochs = 5000
lr = 0.1

#d is a list of layers dimentions
d = [2,2,1]
#f is a list of functions activations(without input layer's activation fun)
f = [sigmoidal, heaviside]
#f_p is a list of f_point(proizvodnyh)
f_p = [sigmoidal_point, heaviside_point]

#creation of Neural Net
nrn = NN( d, f, f_p)
#learning
nrn.learn(lr, epochs, x_arr, bra_y)

#check of workability
print('RESULT: ')
for i in range(x_arr.shape[0]):
    bra_x = np.array(x_arr[i],ndmin=2)
    print(bra_x, ' XOR = ', nrn.make_calculation(bra_x))
    

