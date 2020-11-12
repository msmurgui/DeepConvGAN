import tensorflow as tf
from generator import Generator
from discriminator import Disriminator
import numpy as np

class GAN():
    def __init__(self):
        self.generator = Generator()
        self.discriminator = Disriminator()
        self.testNoises = tf.random.normal([100, 1, 100])
        self.testProgression = []

        self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        self.gOptimizer = tf.keras.optimizers.Adam(1e-4) #try 1e-4
        self.dOptimizer = tf.keras.optimizers.Adam(1e-4)

    def call(self, n):
        return self.generator(n)

    def generatorLoss(self, fakeOutput):
        # Backprop loss. We want to maximize the discriminator's
        # loss, which is equivalent to minimizing the loss with the true
        # labels flipped (i.e. y_true=1 for fake images). We do this
        # as tf can only minimize a function instead of maximizing
        return self.bce(tf.ones_like(fakeOutput), fakeOutput)

    def discriminatorLoss(self, realOutput, fakeOutput):
        # Real images
        real_loss = self.bce(tf.ones_like(realOutput), realOutput)
        # Fake images
        fake_loss = self.bce(tf.zeros_like(fakeOutput), fakeOutput)

        return real_loss + fake_loss

    def applyGenGrads(self, genGradients):
        self.gOptimizer.apply_gradients(zip(genGradients, self.generator.trainable_variables))
    
    def applyDiscGrads(self, discGradients):
        self.dOptimizer.apply_gradients(zip(discGradients, self.discriminator.trainable_variables))
