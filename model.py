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
#def model_fn(features, targets, mode, params):
#        return predictions, loss, train_op


def weibull_loglikelihood(args):
    def loss(y_true, y_pred):
        a_, b_, y_, u_ = args
        hazard0 = tf.pow(tf.div(y_ + 1e-35, a_), b_)
        hazard1 = tf.pow(tf.div(y_ + 1, a_), b_)
        return (tf.mul(u_, tf.log(tf.exp(hazard1 - hazard0) - 1.0)) - hazard1)
    return loss

def weibull_beta_penalty(b_,location = 10.0, growth=20.0, name=None):
    # Regularization term to keep beta below location

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

# # Train model to output the range we want for betas and alphas
# model_initializer = Model(input_data, [alpha, beta])
# model_initializer.compile(loss='mae', optimizer='adam')
# X_temp = np.random.randint(2, size=(5000, timepoints, features))
# alphas_temp = np.random.randint(10,20,(5000,1))
# betas_temp = np.random.randint(10,20,(5000,1))
# model_initializer.fit(X_temp, [alphas_temp, betas_temp], nb_epoch=1)


# Implement custom loss in a lambda layer
# loss_out = Lambda(weibull_loglikelihood, output_shape=(1,), name='WTTE')([alpha, beta, tte, censored])
# model = Model(input=[input_data, tte, censored], output=[loss_out])
model = Model(input=[input_data, tte, censored], output=[alpha, beta])

#loss calculation occurs in lambda layer, so use dummy lambda here for loss
sgd = SGD(lr=0.00001)
# model.compile(loss={'WTTE': lambda y_true, y_pred: y_pred}, optimizer=sgd)
model.compile(loss=weibull_loglikelihood([alpha, beta, tte, censored]), optimizer=sgd)

# weibull_params_func = K.function([input_data], [alpha, beta])
# loss_func = K.function([input_data, tte, censored], [loss_out])


#============================================#
#  Specify input data (X, TTE, and censd)
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
# censd = np.ones(shape=(samples,timepoints))

# Dummy labels for dummy loss function
dummy_labels = np.zeros(shape=(samples, 1))



#============================================#
#                 Train
#============================================#
output = model.predict([X, TTE, censd])
# loss = loss_func([X, TTE, censd])
# alphas, betas = weibull_params_func([X])

model.fit([X, TTE, censd],
          [dummy_labels, dummy_labels],
          nb_epoch=1)

# alphas, betas = weibull_params_func([X])
output = model.predict([X, TTE, censd])
