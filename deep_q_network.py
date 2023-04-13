import numpy as np
import tensorflow as tf

def deep_q_network(state_shape, action_size, learning_rate, hidden_neurons):
    state_input = tf.keras.Input(state_shape, name='frames')

    hidden_1 = tf.keras.layers.Dense(hidden_neurons, activation='relu')(state_input)
    hidden_2 = tf.keras.layers.Dense(hidden_neurons, activation='relu')(hidden_1)
    q_values = tf.keras.layers.Dense(action_size)(hidden_2)

    model = tf.keras.Model(inputs=[state_input], outputs=q_values)
    optimizer = tf.keras.optimizers.RMSprop(lr=learning_rate)
    model.compile(loss='mse', optimizer=optimizer)
    return model