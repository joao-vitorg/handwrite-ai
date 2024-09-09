import keras


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = keras.utils.normalize(x_train, axis=1)
x_test = keras.utils.normalize(x_test, axis=1)

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')]
)

model.summary()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=["accuracy"])

model.fit(x_train, y_train, epochs=15, batch_size=128, validation_split=0.1)

model.save('handwritten1.keras')