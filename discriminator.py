import tensorflow as tf 

class Disriminator(tf.keras.Model):
    def __init__(self):
        super(Disriminator, self).__init__()
        
        self.conv1 = tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding= 'same', input_shape=[28, 28, 1])
        self.leakyRelu1 = tf.keras.layers.LeakyReLU()
        self.drop1 = tf.keras.layers.Dropout(0.3)

        self.conv2 = tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding= 'same')
        self.leakyRelu2 = tf.keras.layers.LeakyReLU()
        self.drop2 = tf.keras.layers.Dropout(0.3)

        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(1)
    
    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.leakyRelu1(x)
        x = self.drop1(x, training= training)

        x = self.conv2(x)
        x = self.leakyRelu2(x)
        x = self.drop2(x, training=training)

        x = self.flatten(x)
        return self.dense(x)
