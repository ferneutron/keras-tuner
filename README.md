# Keras Tuner

This repository contains two implementations on how to use `keras-tuner` for hyperparameter tuning on `keras`-based and `sklearn`-based models.


[![alt text](img/img.jpg)](https://youtu.be/OUOpxEwa2Ek?si=IYk53ItHfjvspod5)


## How to use it?
It is recommended to follow the step by step described below. However, you can adapt the code to your needs.

### Step 1.
Build the docker image as follows:

```bash
$ docker build -t keras-tuner:v1 .
```

### Step 2.
Run the container and access it through the shell. 

**Note**: It is recommended that you mount the current directory so that you have access to the python scripts.
If you want to skip mounting the current directory, you will have to modify `Dockerfile` to add `COPY` commands to move scripts from the image build.

```bash
$ docker run -it -v $PWD:/home/app/ keras-tuner:v1 /bin/bash
```

### Step 3.

#### 3.1 Scikit-learn Optimization

To optimize the sample model provided in the `sklearn_tuning.py`, run:

```bash
$ python -B sklearn_tuning.py
```

#### 3.2 Keras Optimization

To optimize the sample model provided in the `keras_tuning.py`, run:

```bash
$ python -B keras_tuning.py
```

then, for launching the `TensorBoard` visualization, run:

```bash
$ tensorboard --logdir tensorboard
```

**Note**: The script keras_tuning.py contains the variable `KERAS_PROJECT_TENSORBOARD = "tensorboard"`, this is the reason why the `--logdir` is pointing to `tensorboard`.
