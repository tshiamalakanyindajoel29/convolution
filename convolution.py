import argparse
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def apply_convolution(image_path="sample.jpg", output_size=(300, 300), show=True):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Fichier introuvable : {image_path}")

    image_data = tf.io.read_file(image_path)
    try:
        image = tf.io.decode_jpeg(image_data, channels=1)
    except Exception:
        image = tf.io.decode_image(image_data, channels=1, expand_animations=False)
        image = tf.image.resize(image, output_size, method=tf.image.ResizeMethod.BILINEAR)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32) 
        image = tf.expand_dims(image, axis=0)  

    kernel = tf.constant([[-1, -1, -1],
                          [-1, 8, -1],
                          [-1, -1, -1]], dtype=tf.float32)
    kernel = tf.reshape(kernel, [3, 3, 1, 1])  
    conv_output = tf.nn.conv2d(image, filters=kernel, strides=[1, 1, 1, 1], padding='SAME')

    conv_img = tf.squeeze(conv_output, axis=[0, -1]) 
    conv_np = conv_img.numpy()

    min_v, max_v = conv_np.min(), conv_np.max()
    if max_v > min_v:
        conv_norm = (conv_np - min_v) / (max_v - min_v)
    else:
        conv_norm = np.zeros_like(conv_np)

    if show:
        plt.imshow(conv_norm, cmap='gray', vmin=0, vmax=1)
        plt.title("Résultat de la Convolution")
        plt.axis('off')
        plt.show()

    return conv_norm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Appliquer une convolution sur une image.")
    parser.add_argument("image", nargs="?", default="sample.jpg", help="Chemin du fichier image")
    parser.add_argument("--width", type=int, default=300, help="Largeur de redimensionnement")
    parser.add_argument("--height", type=int, default=300, help="Hauteur de redimensionnement")
    parser.add_argument("--no-show", action="store_true", help="Ne pas afficher l'image")
    args = parser.parse_args()

    try:
        apply_convolution(args.image, output_size=(args.height, args.width), show=not args.no_show)
        print("Convolution appliquée avec succès.")
    except Exception as e:
        print(f"Erreur lors de la convolution : {e}")
