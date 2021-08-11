#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 11:53:52 2021

@author: hugo
"""

import tensorflow as tf
import PIL
import numpy as np
import sys

# path de l'image par défaut si aucune image n'est donnée en entrée
image_path = 'P6_03_image_Siberian_husky_wikipedia.jpg'

if (len(sys.argv) > 1):
    image_path = sys.argv[1]

# Chargement et preprocessing de l'image
img = tf.keras.preprocessing.image.load_img(image_path, target_size=(160,160))
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_batch = np.expand_dims(img_array, axis=0)
img_batch_prepro = tf.keras.applications.densenet.preprocess_input(img_batch)

# Chargement du modèle et du dictionnaire des classes
model = tf.keras.models.load_model('results/TF/best_model')
validation_class_indices = np.load('results/TF/best_model/validation_class_indices.npy', allow_pickle=True).item()
results_dict = {indice: label for label, indice in validation_class_indices.items()}

# prediction de l'image avec le modèle
prediction = model.predict(img_batch_prepro)

# affichage de la classe prédite
print(results_dict[prediction.argmax()])