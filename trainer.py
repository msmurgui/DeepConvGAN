from gan import GAN
import tensorflow as tf 
import time
from utils import generateAndSaveImages
import os
from tqdm import tqdm

class Trainer():
    def __init__(self, epochs,batchSize, noiseDim, numExamples2Generate):
        self.epochs = epochs
        self.batchSize = batchSize
        self.noiseDim = noiseDim
        self.numExamples2Generate = numExamples2Generate

        self.model = GAN()


        self.checkpointDir = './trainingCheckpoints'
        self.checkpointPrefix = os.path.join(self.checkpointDir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(gOptimizer=self.model.gOptimizer,
                                 dOptimizer=self.model.dOptimizer,
                                 generator=self.model.generator,
                                 discriminator=self.model.discriminator)

        if(tf.train.latest_checkpoint('./trainingCheckpoints')):
            self.checkpoint.restore(tf.train.latest_checkpoint('./trainingCheckpoints'))
            print('Restored from last checkpoint')

        self.seed = tf.random.normal([numExamples2Generate, noiseDim])



    @tf.function
    def trainStep(self, images):
        noise = tf.random.normal([self.batchSize, self.noiseDim])

        with tf.GradientTape() as genTape, tf.GradientTape() as discTape:
            generatedImages = self.model.generator(noise, training=True)

            realOutput = self.model.discriminator(images, training=True)
            fakeOutput = self.model.discriminator(generatedImages, training=True)

            genLoss = self.model.generatorLoss(fakeOutput)
            discLoss = self.model.discriminatorLoss(realOutput, fakeOutput)
        
        genGradients = genTape.gradient(genLoss, self.model.generator.trainable_variables)
        discGradients = discTape.gradient(discLoss, self.model.discriminator.trainable_variables)
        self.model.applyGenGrads(genGradients)
        self.model.applyDiscGrads(discGradients)

    
    def train(self, dataset):
        for epoch in range(self.epochs):
            start = time.time()
        
            for imageBatch in tqdm(dataset):
                self.trainStep(imageBatch)
            
            #Display images on the go
            generateAndSaveImages(self.model.generator, epoch + 1, self.seed)

            #Save the model every 15 epochs
            if(epoch + 1) % 15 == 0:
                self.checkpoint.save(file_prefix = self.checkpointPrefix)

            print('Time for epoch {} is {} sec'.format( epoch+1 , time.time()-start))
        #Generate after the final epoch
        generateAndSaveImages(self.model.generator, self.epochs, self.seed)
