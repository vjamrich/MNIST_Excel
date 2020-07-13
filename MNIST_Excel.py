import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
from keras import layers
from keras import models
from keras.datasets import mnist
from keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


def plot(digit):
    plt.imshow(digit, cmap=plt.cm.binary)
    plt.show()
    return None


def resize(dgt, res=28, targetres=14):
    newdgt = scipy.ndimage.zoom(dgt, targetres / res, order=0)
    return newdgt


def crop(dgt, cropx, cropy):
    y, x = dgt.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return dgt[starty:starty + cropy, startx:startx + cropx]


network = models.Sequential()
network.add(layers.Dense(128, activation='relu', input_shape=(12 * 12,)))
network.add(layers.Dense(10, activation='softmax'))

network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

train_images = np.array([resize(dgt) for dgt in train_images])
train_images = np.array([crop(dgt, 12, 12) for dgt in train_images])
train_images = train_images.astype('float32') / 255
train_images = np.round(train_images, decimals=0)
train_images = train_images.reshape((60000, 12 * 12))
test_images = np.array([resize(dgt) for dgt in test_images])
test_images = np.array([crop(dgt, 12, 12) for dgt in test_images])
test_images = test_images.astype('float32') / 255
test_images = np.round(test_images, decimals=0)
test_images = test_images.reshape((10000, 12 * 12))

print(train_images.shape)
print(test_images.shape)

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

train_images_val = train_images[:4000]
partial_train_images = train_images[4000:]
train_labels_val = train_labels[:4000]
partial_train_labels = train_labels[4000:]

history = network.fit(partial_train_images,
                      partial_train_labels,
                      epochs=9,
                      batch_size=128,
                      validation_data=(train_images_val, train_labels_val))
results = network.evaluate(test_images, test_labels)

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

epochs = range(1, len(acc_values) + 1)

plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf()
plt.plot(epochs, acc_values, 'bo', label='Training acc')
plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.legend()
plt.show()

weights = network.get_weights()
print(weights)
np.save('Weights_MNIST_Excel.npy', weights)
