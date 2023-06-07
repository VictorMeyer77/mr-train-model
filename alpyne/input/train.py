from tensorflow.python.keras import models, layers

from utils.data_loader import load_dataset
import tensorflow as tf

INPUT_PATH = "/home/victor/Entreprise/Developpement/mr-train-model/alpyne/input/{}"

x_train, y_train, x_test, y_test = load_dataset(INPUT_PATH, ["0", "1", "2", "3"], grayscale=False)
x_train, x_test = x_train / 255.0, x_test / 255.0


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 3), name="test_in"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(4, name="test_out"))

model.summary()


loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer="adam",
              loss=loss_fn,
              metrics=["accuracy", "sparse_categorical_crossentropy"])

model.fit(x_train, y_train, epochs=5, batch_size=100, validation_data=(x_test, y_test))

model.save("test_model", save_format="tf")
