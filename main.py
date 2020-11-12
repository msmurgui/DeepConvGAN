import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_transform as tft
from generator import Generator
from discriminator import Disriminator
import matplotlib.pyplot as plt
import glob

from trainer import Trainer

BUFFER_SIZE = 60000
BATCH_SIZE = 256
EPOCHS = 50
noise_dim = 100
num_examples_to_generate = 16


def getData():
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    train_images = (train_images - 127.5) / 127.5 # Normalize the images to [-1, 1]
    # Batch and shuffle the data
    trainDataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    return trainDataset


trainer = Trainer(EPOCHS, BATCH_SIZE, noise_dim, num_examples_to_generate)

dataset = getData()
trainer.train(dataset)