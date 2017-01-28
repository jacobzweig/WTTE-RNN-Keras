#Change directory to file path
import os
os.chdir('/Users/jacobzweig/Documents/Strong/strong-internal/WTTE-RNN/')

import tensorflow as tf
import numpy as np
import keras 
from keras import backend as K
from keras.layers import Input, LSTM, Dense, Lambda
from keras.callbacks import LambdaCallback
from keras.models import Model
from keras.optimizers import SGD

elu = keras.layers.advanced_activations.ELU(alpha=1.0)

sess = tf.Session()
K.set_session(sess)

samples = 2000
timepoints = 100
spacing = 5
features = 2

#===========================================================================================#
# Define Model with alpha (exponential activation) & beta (softplus activation) as outputs
#===========================================================================================#

# We put a wrapper around the loss function since keras wants args to be (y_true, y_pred)
# We could alternatively implement this as a lambda layer, but this seemed easier
def weibull_loss(args):
    def loss(y_true, y_pred):
        a_, b_, y_, u_ = args
        hazard0 = tf.pow(tf.div(y_ + K.epsilon(), a_), b_)
        hazard1 = tf.pow(tf.div(y_ + 1, a_), b_)
        loglikelihood = (tf.mul(u_, tf.log(tf.exp(hazard1 - hazard0) - 1.0)) - hazard1)
        return -tf.reduce_mean(loglikelihood)
    return loss

def weibull_beta_penalty(b_,location = 10.0, growth=20.0, name=None):
    # Regularization term to keep beta below location
    # Not using this yet

    with tf.name_scope(name):
        scale = growth/location
        penalty_ = tf.exp(scale*(b_-location))
    return(penalty_)

tte = Input(shape=[1], name="Time_to_event")
censored = Input(shape=[1], name="Censored")
input_data = Input(shape=(timepoints, 2), name="Input_data")
x = LSTM(32, activation='relu', return_sequences=True)(input_data)
x = LSTM(32, activation='relu')(x)
alpha = Dense(1, activation='softplus')(x)
beta = Dense(1, activation='softplus')(x)

model = Model(input=[input_data, tte, censored], output=[alpha, beta])

sgd = SGD(lr=0.0001)
model.compile(loss=weibull_loss([alpha, beta, tte, censored]), optimizer=sgd)



#============================================#
#  Specify input data (X, TTE, and censd)
#     This needs a lot of cleaning up
#============================================#

TTE = np.zeros(shape=(samples, timepoints))
for sample in range(samples):
    counter = np.random.randint(1, 25)
    for timepoint in range(timepoints):
        TTE[sample,timepoint] = counter
        counter -= 1
        if counter < 0:
            counter = spacing
    
# Set features after event to 1
X = np.zeros(shape=(samples, timepoints, 2))
for sample in range(TTE.shape[0]):
    for timepoint in range(TTE.shape[1]):
        if TTE[sample, timepoint] == 1:
            try:
                X[sample,timepoint+1,:] = 1.0
            except: # Catch timepoints at end
                pass

TTE = TTE[:,1].reshape(samples,1)
censd = np.random.randint(2, size=(samples, 1))

# Dummy labels for dummy loss function
dummy_labels = np.zeros(shape=(samples, 1))



#============================================#
#                 Train
#============================================#
output = model.predict([X, TTE, censd])

model.fit([X, TTE, censd],
          [dummy_labels, dummy_labels],
          nb_epoch=5)

# alphas, betas = weibull_params_func([X])
output = model.predict([X, TTE, censd])

model.fit([X, TTE, censd],
          [dummy_labels, dummy_labels],
          nb_epoch=1)