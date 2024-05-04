import numpy as np
import keras_tuner
import keras
from keras import layers

KERAS_TRIALS_DIR = "."
KERAS_PROJECT_NAME = "keras_hp"
KERAS_PROJECT_TENSORBOARD = "tensorboard"

def build_model(hp):
    
    model_type = hp.Choice("model_type", ["mlp", "cnn"])

    inputs = keras.Input(shape=(28, 28, 1))
    x = inputs

    match model_type:
        case "mlp":
            x = layers.Flatten()(x)
            for layer in range(hp.Int("mlp_layers", 1, 3)):
                x = layers.Dense(
                    units=hp.Int(f"units_{layer}", 32, 128, step=32),
                    activation=hp.Choice("activation", values=["relu", "tanh"]),
                )(x)

        case "cnn":
            for layer in range(hp.Int("cnn_layers", 1, 3)):
                x = layers.Conv2D(
                    hp.Int(f"filters_{layer}", 32, 128, step=32),
                    kernel_size=(3, 3),
                    activation=hp.Choice("activation", values=["relu", "tanh"]),
                )(x)
                x = layers.MaxPooling2D(pool_size=(2, 2))(x)
            x = layers.Flatten()(x)

        case _:
            return None

    if hp.Boolean("dropout"):
        x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(units=10, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
        optimizer="adam",
    )
    return model


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    tuner = keras_tuner.RandomSearch(
        build_model,
        max_trials=10,
        objective="val_accuracy",
        directory=KERAS_TRIALS_DIR,
        project_name=KERAS_PROJECT_NAME
    )

    tuner.search(
        x_train,
        y_train,
        validation_split=0.2,
        epochs=2,
        callbacks=[keras.callbacks.TensorBoard(KERAS_PROJECT_TENSORBOARD)],
    )