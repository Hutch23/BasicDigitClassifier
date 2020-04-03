# -*- coding: utf-8 -*-
"""
Created on Fri Apr 3 10:21:31 2020

@author: Parker Hutchinson

Some inspiration from:
https://www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

""" LOAD DATA """
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

"""
# Display sample input
plt.imshow(x_train[0], cmap='gray')
plt.show()
"""

""" DATA PREPROCESSING """
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

""" MODEL CREATION """
hidden_layer_size = 128
output_size = 10
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape = (28, 28)),
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
    tf.keras.layers.Dense(output_size, activation='softmax')
    ])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

model.fit(x=x_train, y=y_train, epochs=10)

""" MODEL EVALUATION """
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test set Loss: {0} \tAccuracy: {1:.2f}%'.format(test_loss, test_acc*100))

from random import random

def print_predictions(predictions, verbose=1):
    prediction_digit = np.argmax(predictions)
    confidence_percentage = np.max(predictions) * 100
    print('Prediction: {0} with {1:.2f}% confidence'.format(prediction_digit, 
                                                        confidence_percentage))
    
    if (verbose == 2):
        print('Digit:\t Confidence')
        for digit, confidence in enumerate(predictions[0]):
            print('{0}:\t {1:.2f}'.format(digit, confidence*100))
    
def test_model_descriptive(test_input, verbose=1, actual=None):
    # Plot the input image
    plt.imshow(test_input, cmap='gray')
    plt.show()
    # Calculate and display predictions
    predictions = model.predict(test_input.reshape(1, 28, 28))
    print_predictions(predictions, verbose)
    # If a correct answer is provided, display it
    if actual is not None:
        print('Actual digit:', actual)
        
    
def test_model_on_random():
    input_index = int(random()*len(x_test))
    test_model_descriptive(x_test[input_index], actual=y_test[input_index])
    

test_model_on_random()
prompt = 'Press enter to see more model predictions on random test examples,' \
    ' or q to quit'
answer = input(prompt)
while answer.lower() != 'q':
    test_model_on_random()
    answer = input(prompt)
    
""" CUSTOM IMAGE INPUT """
import imageio

def get_image(filename):
    im = imageio.imread(filename) 
    
    grayscale_image = np.dot(im[...,:3], [0.299, 0.587, 0.114])
    
    """
    The input should be a white digit on a black background. By default, lower
    values in the image indicate darkness (255 is white), so if the average value
    of the image is greater than 255 / 2 (most of the image contains white),
    invert the image colors.
    """
    if np.average(grayscale_image) > 255 / 2:
        grayscale_image = abs(grayscale_image - 255)
    
    normalized_image = tf.keras.utils.normalize(grayscale_image, axis=1)
    
    return normalized_image

test_model_descriptive(get_image('handwritten2.png'))
