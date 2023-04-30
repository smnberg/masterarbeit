
import numpy as np
import matplotlib.pyplot as plt

#import tensorflow_datasets as tfds
import tensorflow as tf





import attention_data_generator as adg
from keras import backend as K


#for comparisons
def run_dense():

    n_visits = 30
    encoding_dim = 10

    
    input_layer = tf.keras.layers.Input(shape = (n_visits ,encoding_dim))


    flattened = tf.keras.layers.Flatten()( input_layer )
    
    dense1 = tf.keras.layers.Dense(15, use_bias=False, activation = "ReLU")(flattened)
    dense2 = tf.keras.layers.Dense(10, use_bias=False, activation = "ReLU")(dense1)

    norm_layer = tf.keras.layers.Normalization() (dense2)

    dense3 = tf.keras.layers.Dense(2, use_bias=False) (norm_layer)
    output_layer = tf.keras.activations.softmax(dense3)


    model = tf.keras.Model(input_layer, output_layer)
    model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(),
              optimizer = tf.optimizers.Adam(),
              metrics   = [tf.keras.metrics.SparseCategoricalAccuracy()])

    return model




def linformer_single_head_attention():
    n_visits = 30
    encoding_dim = 10
    proj_dim = 25

    input_layer = tf.keras.layers.Input(shape = (n_visits ,encoding_dim))
    key_layer   = tf.keras.layers.Dense(4, use_bias=False)(input_layer) #Using ReLU seems to minimally better the result
    queue_layer = tf.keras.layers.Dense(4, use_bias=False)(input_layer)
    value_layer = tf.keras.layers.Dense(4, use_bias=False)(input_layer)

        
    random_proj_matrix_value    = np.random.normal(0, 1, [proj_dim, n_visits])
    random_proj_matrix_key      = np.random.normal(0, 1, [proj_dim, n_visits])
    
    proj_key_layer              = tf.keras.layers.Lambda(lambda x: tf.linalg.matmul(x[0], x[1] )) ([random_proj_matrix_key, key_layer])
    proj_value_layer            = tf.keras.layers.Lambda(lambda x: tf.linalg.matmul(x[0], x[1] )) ([random_proj_matrix_value, value_layer])

    permute_layer = tf.keras.layers.Permute((2,1), input_shape=(proj_dim, 4)) (proj_key_layer)
    
    matrix_mul_layer = tf.keras.layers.Lambda(lambda x: tf.linalg.matmul(x[0], x[1] )) ([queue_layer, permute_layer])

    softmax_layer = tf.keras.layers.Softmax(axis=2) (matrix_mul_layer)

    matrix_mul_value_weighting_layer = tf.keras.layers.Lambda(lambda x: tf.linalg.matmul(x[0], x[1])) ([softmax_layer, proj_value_layer])

    norm_layer = tf.keras.layers.Normalization() (matrix_mul_value_weighting_layer)

    
    flatten_layer = tf.keras.layers.Flatten()(norm_layer)


    dense_layer_3 = tf.keras.layers.Dense(2, use_bias=False)(flatten_layer)


    output_layer = tf.keras.activations.softmax(dense_layer_3)

    model = tf.keras.Model(input_layer, output_layer)

    model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(),
              optimizer = tf.optimizers.Adam(),
              metrics   = [tf.keras.metrics.SparseCategoricalAccuracy()])

    
    return model



def single_head_attention():

    n_visits = 30
    encoding_dim = 10
    performer = True
    
    input_layer = tf.keras.layers.Input(shape = (n_visits ,encoding_dim))
    #tmp = DebugLayer('at input')(input_layer)
    
    #encod = positional_encoding(8,6)

    #encoding_layer = tf.keras.layers.Add() ([input_layer, encod])

    #flatten_layer = tf.keras.layers.Flatten()(encoding_layer)
  
    key_layer   = tf.keras.layers.Dense(4, use_bias=False)(input_layer) #Using ReLU seems to minimally better the result
    queue_layer = tf.keras.layers.Dense(4, use_bias=False)(input_layer)
    value_layer = tf.keras.layers.Dense(4, use_bias=False)(input_layer)


    
    permute_layer = tf.keras.layers.Permute((2,1), input_shape=(n_visits, 4)) (key_layer)
    
    matrix_mul_layer = tf.keras.layers.Lambda(lambda x: tf.linalg.matmul(x[0], x[1] )) ([queue_layer, permute_layer])

    
    softmax_layer = tf.keras.layers.Softmax(axis=2) (matrix_mul_layer)

  
   
    matrix_mul_value_weighting_layer = tf.keras.layers.Lambda(lambda x: tf.linalg.matmul(x[0], x[1])) ([softmax_layer, value_layer])

    

    norm_layer = tf.keras.layers.Normalization() (matrix_mul_value_weighting_layer)


    
    flatten_layer = tf.keras.layers.Flatten()(norm_layer)


    dense_layer_3 = tf.keras.layers.Dense(2, use_bias=False)(flatten_layer)

    output_layer = tf.keras.activations.softmax(dense_layer_3)

    model = tf.keras.Model(input_layer, output_layer)

    model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(),
              optimizer = tf.optimizers.Adam(),
              metrics   = [tf.keras.metrics.SparseCategoricalAccuracy()])

    
    return model







def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
    pos_encoding = angle_rads[np.newaxis, ...]
    
    return tf.cast(pos_encoding, dtype=tf.float32)




if __name__== "__main__":
    
    
    n_training_samples = 100000 #sample sizes have to be dividable by the batch_size because of the very flimsy implementation of the classification layer of multi_head_attention
    n_test_samples = 96

    training_x, training_y_tmp = adg.create_data_record(n_training_samples)
    
    validation_x, validation_y = adg.create_data_record(n_test_samples)
    
    #comparison of linformer and stanard attention.
    
    model_1 = single_head_attention()

    history_1 = model_1.fit(training_x, training_y_tmp, epochs=40)

    model_2 = linformer_single_head_attention()
    history_2 = model_2.fit(training_x, training_y_tmp, epochs=40)

    plt.plot(history_1.history['sparse_categorical_accuracy'])
    plt.plot(history_2.history['sparse_categorical_accuracy'])
    plt.title('model accuracy, k=25, n=30,samples=100000')
    plt.ylabel('sparse_categorical_accuracy')
    plt.xlabel('epoch')
    plt.legend(['standard_attention', 'linformer'], loc='upper left')
    plt.show()
