import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

# ----- Model 1 (input layer -> 64 layer -> 64 layer) -----
model1 = models.Sequential()
model1.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model1.add(layers.MaxPooling2D((2, 2)))
model1.add(layers.Conv2D(64, (3, 3), activation='relu'))
model1.add(layers.MaxPooling2D((2, 2)))
model1.add(layers.Conv2D(64, (3, 3), activation='relu'))
model1.add(layers.Flatten())
model1.add(layers.Dense(64, activation='relu'))
model1.add(layers.Dense(10))
model1.summary()

model1.compile(optimizer='adam',
               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
               metrics=['accuracy'])

history1 = model1.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

# ----- Model 2 (input layer -> 64 layer -> 128 layer) -----
model2 = models.Sequential()
model2.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model2.add(layers.MaxPooling2D((2, 2)))
model2.add(layers.Conv2D(64, (3, 3), activation='relu'))
model2.add(layers.MaxPooling2D((2, 2)))
model2.add(layers.Conv2D(128, (3, 3), activation='relu'))
model2.add(layers.Flatten())
model2.add(layers.Dense(128, activation='relu'))
model2.add(layers.Dense(10))
model2.summary()

model2.compile(optimizer='adam',
               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
               metrics=['accuracy'])

history2 = model2.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

# ----- Model 3 (input layer -> 128 layer -> 128 layer) -----
model3 = models.Sequential()
model3.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model3.add(layers.MaxPooling2D((2, 2)))
model3.add(layers.Conv2D(128, (3, 3), activation='relu'))
model3.add(layers.MaxPooling2D((2, 2)))
model3.add(layers.Conv2D(128, (3, 3), activation='relu'))
model3.add(layers.Flatten())
model3.add(layers.Dense(128, activation='relu'))
model3.add(layers.Dense(10))
model3.summary()

model3.compile(optimizer='adam',
               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
               metrics=['accuracy'])

history3 = model3.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

# ----- Plotting all three models' accuracy and validation accuracy -----
plt.figure(figsize=(12, 8))
plt.plot(history1.history['accuracy'], label='Model1 Train Accuracy')
plt.plot(history1.history['val_accuracy'], label='Model1 Val Accuracy')
plt.plot(history2.history['accuracy'], label='Model2 Train Accuracy')
plt.plot(history2.history['val_accuracy'], label='Model2 Val Accuracy')
plt.plot(history3.history['accuracy'], label='Model3 Train Accuracy')
plt.plot(history3.history['val_accuracy'], label='Model3 Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.title('Comparison of Model Accuracies')
plt.legend(loc='lower right')
plt.show()