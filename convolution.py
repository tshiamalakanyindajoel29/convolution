import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def apply_convolution():
    try:
        image_path = r'C:\dossier joel\_DSC5064.jpg'
        image = tf.io.read_file(image_path)
        image = tf.io.decode_jpeg(image, channels=1)
        image = tf.image.resize(image, [300, 300])
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.expand_dims(image, axis=0)
        kernel = tf.constant([[-1, -1, -1],
                              [-1, 8, -1],
                              [-1, -1, -1]], dtype=tf.float32)
        kernel = tf.reshape(kernel, [3, 3, 1, 1])
        conv_output = tf.nn.conv2d(image, filters=kernel, strides=1, padding='SAME')
        plt.imshow(tf.squeeze(conv_output), cmap='gray')
        plt.title('Résultat de la Convolution')
        plt.axis('off')
        plt.show()
        print(" Convolution appliquée avec succès.")
    except Exception as e:
        print(f" Erreur lors de l'application de la convolution : {e}")