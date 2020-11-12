import tensorflow as tf 

class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        self.firstCall = True

        self.dense1 = tf.keras.layers.Dense(7*7*256, use_bias=False, input_shape=(100,))
        self.batchNorm1 = tf.keras.layers.BatchNormalization()
        self.leakyRelu1 = tf.keras.layers.LeakyReLU()

        self.reshape1 = tf.keras.layers.Reshape((7,7,256))
        
        self.conv1 = tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False)
        self.batchNorm2 = tf.keras.layers.BatchNormalization()
        self.leakyRelu2 = tf.keras.layers.LeakyReLU()

        self.conv2 = tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)
        self.batchNorm3 = tf.keras.layers.BatchNormalization()
        self.leakyRelu3 = tf.keras.layers.LeakyReLU()

        self.conv3 = tf.keras.layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation=tf.keras.activations.tanh)
    
    def call(self, x, training=False):
        x = self.dense1(x)
        x = self.batchNorm1(x, training=training)
        x = self.leakyRelu1(x)

        x = self.reshape1(x)
        #if(not self.firstCall): assert self.output_shape == (None, 7, 7, 256) #(except of None, 1) ???

        x = self.conv1(x)
        #if(not self.firstCall): assert self.output_shape == (None, 7, 7, 128)

        x = self.batchNorm2(x, training=training)
        x = self.leakyRelu2(x)

        x = self.conv2(x)
        #if(not self.firstCall): assert self.output_shape == (None, 14, 14, 64)

        x = self.batchNorm3(x, training=training)
        x = self.leakyRelu3(x)

        x = self.conv3(x)
        #if(not self.firstCall): assert self.output_shape == (None, 28, 28, 1)
        #self.firstCall= False
        return x
    
