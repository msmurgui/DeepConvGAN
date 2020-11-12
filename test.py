import tensorflow as tf 
from generator import Generator
from gan import GAN
import matplotlib.pyplot as plt

model = GAN()
chkpt = tf.train.Checkpoint(gOptimizer=model.gOptimizer,
                                 dOptimizer=model.dOptimizer,
                                 generator=model.generator,
                                 discriminator=model.discriminator)
chkpt.restore(tf.train.latest_checkpoint('./trainingCheckpoints'))

noise = tf.random.normal([256, 100])

generated = model.generator(noise)
plt.imshow(generated[0,:,:,0], cmap='gray')
plt.show()


