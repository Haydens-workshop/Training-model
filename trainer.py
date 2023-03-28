import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Load MNIST
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess
x_train = x_train / 255.0
x_test = x_test / 255.0

# Define
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax')
])

# Compile
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

# Data augmentation
datagen = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)

datagen.fit(x_train[..., np.newaxis])

# Callback
early_stopping = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
model_checkpoint = keras.callbacks.ModelCheckpoint("mnist_best_model.h5", save_best_only=True)

# Train model
history = model.fit(datagen.flow(x_train[..., np.newaxis], y_train, batch_size=32),
                    epochs=50, validation_data=(x_test[..., np.newaxis], y_test),
                    callbacks=[early_stopping, model_checkpoint])

# Evaluation model
test_loss, test_acc = model.evaluate(x_test[..., np.newaxis], y_test, verbose=2)
print('Test accuracy:', test_acc)

# Load saved model
loaded_model = keras.models.load_model("mnist_best_model.h5")

#predictions w/ loaded model
predictions = loaded_model.predict(x_test[..., np.newaxis])
predicted_classes = np.argmax(predictions, axis=1)

# Display confusion matrix
conf_mat = confusion_matrix(y_test, predicted_classes)
print("Confusion Matrix:")
print(conf_mat)

# Plot
plt.matshow(conf_mat, cmap=plt.cm.gray)
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.show()
