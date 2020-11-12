import matplotlib.pyplot as plt
from IPython import display

def generateAndSaveImages(model, epoch, testInput, displayImages=False):
    display.clear_output(wait=True)
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(testInput, training=False)

    fig = plt.figure(figsize=(4,4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig('./results/image_at_epoch_{:04d}.png'.format(epoch))
    if displayImages: 
      plt.show()