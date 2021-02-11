# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 19:07:07 2020

@author: yousef21

This file contains implementation of two custom layers:
1- Delta_activation layer: this layer should sit after the activation layer to optimize the quantization step to high amount of sparsity when inference is delta-based
2- L1 regularization layer: this is a state-less replacement of the delta_activation layer, can be easily exchange with that. This layer only add L1 regularization to the loss function to increase activation sparsity in a more conventional way 
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np


#%%
@tf.custom_gradient
def round_custom(x):
  def grad(dy):
    return dy
  return tf.round(x), grad

@tf.custom_gradient
def no_grad_custom(x):
  def grad(dy):
    return dy * 0.0
  return x, grad

@tf.custom_gradient
def neg_grad_custom(x):
  def grad(dy):
    return dy * (-1.0)
  return x, grad

def quantize(x, q):
  return tf.multiply(round_custom(tf.divide(x, no_grad_custom(q))), neg_grad_custom(q))

class delta_activation(keras.layers.Layer): 
    '''
    Arguments:
    threshold_level='channel-wise', the granuality of qualtization level (threshold), can be channel-wise, neuron-wise or layer-wise
    sp_rate=1.0,  Sparsity factor for the loss function of sparsity, increasing this factor will push toward more sparsity with the cost of accuracy
    n_outputs=1.0, Number of outputs used in the same way as sparsity factor to increase this factor for high fan-out neurons
    thr_init=1e-1, Initial value of quantization level (threshold)
    trainabale_bias=False, Add traininable bias parameters (in the case of delta procesing, the linear operations, like conv and dense should not contain bias)
    activation=None, Define the Activation function here, currently only 'relu' and 'softmax' is added
    max_pooling=None, Define the max_pool kernel size if required
    sigma =False, If you don't want the layer to perform integration operation (sigma) leave this to False
    delta_level=0, If you don't want the layer to perform delta operation (delta) leave this to 0, 1 is when normal delta out is required. 2 is when the first frame is not required (DVS type without backgound)
    name='delta_activation'
    layer_name=None, 
    show_metric=False
    '''
    def __init__(self, threshold_level='channel-wise', sp_rate=1.0, n_outputs=1.0, thr_init=1e-1, trainabale_bias=False, activation=None, max_pooling=None, sigma =False, delta_level=0, show_metric=False, name='delta_activation', thr_trainable=True):
        super(delta_activation, self).__init__(name=name)
        self.rate = sp_rate
        self.n_outputs = n_outputs
        self.thr_init = thr_init
        self.layer_n = name
        self.threshold_level = threshold_level
        self.trainabale_bias = trainabale_bias
        self.activation = activation
        self.max_pooling = max_pooling
        self.sigma = sigma
        self.delta_level = delta_level
        self.show_metric = show_metric
        self.thr_trainable = thr_trainable
        
    def build(self, input_shape):    #input shape is (batch,time,x,y,c)
        #One threshold per neuron
        if(self.threshold_level=='neuron-wise'):
          self.threshold = self.add_weight(shape=input_shape[2:], initializer=keras.initializers.Constant(value=self.thr_init), trainable=self.thr_trainable) #threshold shape: (x,y,c)
        
        #One threshold per channel
        if(self.threshold_level=='channel-wise'):
          self.threshold = self.add_weight(shape=input_shape[-1], initializer=keras.initializers.Constant(value=self.thr_init), trainable=self.thr_trainable) #threshold shape: (c)

        #One threshold per layer
        if(self.threshold_level=='layer-wise'):
          self.threshold = self.add_weight(shape=1, initializer=keras.initializers.Constant(value=self.thr_init), trainable=self.thr_trainable) #threshold shape: 1

        #Make trainable bias values (since bias should be inside this layer)
        if(self.trainabale_bias == True):
          self.bias = self.add_weight(shape=input_shape[-1], initializer=keras.initializers.Constant(value=0), trainable=True) #threshold shape: (c)


    def call(self, inputs, layer_name=None):
        threshold = tf.math.abs(self.threshold)

        #Accumulation  
        if self.sigma==True:
          neuron_states = tf.cumsum(inputs, axis=1)  #cumulative sum over time
        else:
          neuron_states = inputs
        
        if(self.trainabale_bias == True):
          neuron_states = neuron_states + self.bias

        #Activation function
        if(self.activation=='relu'):
            output = quantize(tf.keras.activations.relu(neuron_states), threshold)
        if(self.activation=='softmax'):
            output = quantize(tf.keras.activations.softmax(neuron_states), threshold)
        if(self.activation==None):
            output = quantize(neuron_states, threshold)

        if(self.max_pooling !=None):
            output = tf.nn.max_pool(output, self.max_pooling, strides=1, padding='SAME', data_format=None, name=None)

        #Delta calculation
        # Make the old output (with a blank first frame)  [same as output with a time shift]
        # To measure number of spikes, we do not consider the first wave of spikes from blank frame
        if len(output.shape)==5: #NNtype=='cnn'  
          spikes = tf.subtract(output[:,:-1,:,:,:], output[:,1:,:,:,:])
        if len(output.shape)==3: #NNtype=='fc'  
          spikes = tf.subtract(output[:,:-1,:], output[:,1:,:])  
        
        if(self.delta_level==2): #no background frame
          if len(output.shape)==5: output  = tf.concat([spikes[:,0:1,:,:,:],spikes],1)
          if len(output.shape)==3: output  = tf.concat([spikes[:,0:1,:],spikes],1)
        if(self.delta_level==1): #with background frame (frame[0])
          if len(output.shape)==5: output_old = tf.concat([tf.zeros_like(output[:,0:1,:,:,:]),output[:,:-1,:,:,:]],1)
          if len(output.shape)==3: output_old = tf.concat([tf.zeros_like(output[:,0:1,:]),output[:,:-1,:]],1)
          output  = tf.subtract(output, output_old)
        
        n_frames = tf.cast(tf.math.multiply(tf.shape(spikes)[0],tf.shape(spikes)[1]), dtype=tf.float32) 
        spikes_L0 = tf.cast(tf.math.count_nonzero(spikes), dtype=tf.float32) 
        spikes_L1 = tf.math.reduce_sum(tf.math.abs(spikes))

        n_neurons = tf.cast(tf.math.reduce_prod(tf.shape(spikes)), dtype=tf.float32) 
        spikes_L1_normalized = tf.divide(spikes_L1, n_neurons) 
        #spikes_L0_normalized = tf.divide(spikes_L0, n_neurons)

        self.add_loss(self.rate * self.n_outputs * spikes_L1_normalized) #L1 loss weighted with n_operations
        if self.show_metric:
          self.add_metric(tf.math.divide(spikes_L0,n_frames), name='n_spikes_'+self.layer_n, aggregation='mean') #moving average number of spikes
          self.add_metric(tf.math.divide(n_neurons,n_frames), name='n_neurons_'+self.layer_n, aggregation='mean') #number of neurons
          self.add_metric(self.n_outputs, name='n_outputs_'+self.layer_n, aggregation='mean') #number of outputs
        return output, spikes_L0, n_neurons


#a layer to increase spatial sparsity only (stateless)
class L1_regulizer(keras.layers.Layer): 
    def __init__(self, sp_rate=1.0, n_outputs=1.0, name='L1_regulizer', show_metric=False, thr_init=None ): 
        super(L1_regulizer, self).__init__(name=name)
        self.n_outputs = n_outputs
        self.layer_n = name
        self.rate = sp_rate
        self.show_metric = show_metric

    def call(self, inputs, layer_name=None):
        output = inputs
        spikes = inputs 

        n_frames = tf.cast(tf.math.multiply(tf.shape(spikes)[0],tf.shape(spikes)[1]), dtype=tf.float32) 
        spikes_L0 = tf.cast(tf.math.count_nonzero(spikes), dtype=tf.float32) 
        spikes_L1 = tf.math.reduce_sum(tf.math.abs(spikes))

        n_neurons = tf.cast(tf.math.reduce_prod(tf.shape(spikes)), dtype=tf.float32) 
        spikes_L1_normalized = tf.divide(spikes_L1, n_neurons) 
        # spikes_L0_normalized = tf.divide(spikes_L0, n_neurons)
        
        self.add_loss(self.rate * self.n_outputs * spikes_L1_normalized) #L1 loss weighted with n_operations
        if self.show_metric:
          self.add_metric(tf.math.divide(spikes_L0,n_frames), name='n_spikes_'+self.layer_n, aggregation='mean') #moving average number of spikes
          self.add_metric(tf.math.divide(n_neurons,n_frames), name='n_neurons_'+self.layer_n, aggregation='mean') #number of neurons
          self.add_metric(self.n_outputs, name='n_outputs_'+self.layer_n, aggregation='mean') #number of outputs

        return output, spikes_L0, n_neurons

















#%%  temporary tests for delta_activation layer
# delta_layer = delta_activation(sp_rate=1e-3, n_outputs=1.0)
# x1 = np.array([[-1.1,  1.1, -1.1,  3.1],
#                 [-1.1,  0.1, -2.1,  0.1],
#                 [-0.1,  1.1, -1.1,  0.1],
#                 [-0.1,  0.1, -1.1,  1.1]],
#               dtype="float32")   #tf.random.normal(shape = (4,4), dtype="float32")
# x1 = np.expand_dims(x1, axis=0)
# y1 = delta_layer(x1, type='fc')
# print(y1)
# print(delta_layer.losses)
      
# delta_layer.threshold.assign_add(tf.ones_like(delta_layer.threshold)*0.4)
    
# x2 = np.array([[-0.45,  3.2, -1.2,  1.2],
#                 [-0.4,  0.2, -2.2,  0.2],
#                 [-0.3 , 1.2, -1.2,  0.3],
#                 [-0.2,  0.2, -1.2,  1.2]],
#               dtype="float32")   #tf.random.normal(shape = (4,4), dtype="float32")  
# x2 = np.expand_dims(x2, axis=0)

# y2 = delta_layer(x2, type='fc')  
# print(y2)     
# print(delta_layer.losses)          
# print(delta_layer.metrics[0].result())
   
##Test for granient 
# delta_layer = delta_activation(sp_rate=1, n_outputs=1.0)

# x = tf.constant(3.0)
# x = tf.expand_dims(x,axis=0)
# x = tf.expand_dims(x,axis=0)
# x = tf.expand_dims(x,axis=0)

# with tf.GradientTape() as tape:
#   y = delta_layer(x, type='fc')

# print(tape.gradient(y, delta_layer.threshold))
# print(tape.gradient(y, x))








































        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        