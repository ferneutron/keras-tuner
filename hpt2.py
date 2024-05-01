import numpy as np
import keras_tuner
import keras
from keras import layers
import tensorflow as tf

from kerastuner_tensorboard_logger import (
    TensorBoardLogger
)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
# Normalize the pixel values to the range of [0, 1].
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Add the channel dimension to the images.
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
# Print the shapes of the data.
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


def build_model(hp):
    inputs = keras.Input(shape=(28, 28, 1))
    # Model type can be MLP or CNN.
    model_type = hp.Choice("model_type", ["mlp", "cnn"])
    x = inputs
    if model_type == "mlp":
        x = layers.Flatten()(x)
        # Number of layers of the MLP is a hyperparameter.
        for i in range(hp.Int("mlp_layers", 1, 3)):
            # Number of units of each layer are
            # different hyperparameters with different names.
            x = layers.Dense(
                units=hp.Int(f"units_{i}", 32, 128, step=32),
                activation="relu",
            )(x)
    else:
        # Number of layers of the CNN is also a hyperparameter.
        for i in range(hp.Int("cnn_layers", 1, 3)):
            x = layers.Conv2D(
                hp.Int(f"filters_{i}", 32, 128, step=32),
                kernel_size=(3, 3),
                activation="relu",
            )(x)
            x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Flatten()(x)

    # A hyperparamter for whether to use dropout layer.
    if hp.Boolean("dropout"):
        x = layers.Dropout(0.5)(x)

    # The last layer contains 10 units,
    # which is the same as the number of classes.
    outputs = layers.Dense(units=10, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)

    # Compile the model.
    model.compile(
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
        optimizer="adam",
    )
    return model


hist_callback = tf.keras.callbacks.TensorBoard(
    log_dir="logsv3/logs")


tuner = keras_tuner.BayesianOptimization(
    build_model,
    max_trials=10,
    # Do not resume the previous search in the same directory.
    overwrite=True,
    objective="val_accuracy",
    # Set a directory to store the intermediate results.
    directory="logsv4",
    project_name='hp'
)


tuner.search(
    x_train,
    y_train,
    validation_split=0.2,
    epochs=2,
    # Use the TensorBoard callback.
    # The logs will be write to "/tmp/tb_logs".
    # callbacks=[keras.callbacks.TensorBoard("logs/hpv2")],
    logger=TensorBoardLogger(
        metrics=["accuracy"], logdir="logsv4/hparams"
    ),  # add only this argument
)





    # Initialize the `HyperParameters` and set the values.
# hp = keras_tuner.HyperParameters()
#hp.values["model_type"] = "cnn"
# Build the model using the `HyperParameters`.
#model = build_model(hp)
# Test if the model runs with our data.
#model(x_train[:100])
# Print a summary of the model.
#model.summary()

# Do the same for MLP model.
#hp.values["model_type"] = "mlp"
#model = build_model(hp)
#model(x_train[:100])
#model.summary()


