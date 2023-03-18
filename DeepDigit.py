import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

#load dataset MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

#preprocess
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0


model = keras.Sequential(
    [
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(10, activation="softmax"),
    ]
)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

#eval model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc}")

n = 25  #number of images
plt.figure(figsize=(10, 10))
for i in range(n):
    ax = plt.subplot(1, n, i + 1)
    img = x_test[i]
    plt.imshow(img, cmap="gray")
    plt.axis("off")
plt.show()