# Cats-Vs-Dogs

### Overview

An Implementation of the [Dogs Vs. Cats](https://www.kaggle.com/competitions/dogs-vs-cats/overview) challenge.  

This project implements three models: A model with 3 convolutional layers which demonstrates overfitting, a model with 3 convolutional and dropout layers to address the previous overfitting, and a model which utilizes a pre-trained MobileNetV2 network as its encoder.

### Data

The dataset can be downloaded from [https://www.kaggle.com/competitions/dogs-vs-cats/overview](https://www.kaggle.com/competitions/dogs-vs-cats/overview).  

The script expects that your data is located in a folder called `train` which should be located at the root of your project. You should then categorize your dataset into the subfolders `cat` and `dog`.   

This can be accomplished on MacOS and Linux by running the following bash command inside the `train` directory which should contain your uncategorized data:

```
mkdir cat && mkdir dog && mv cat.*.jpg cat && mv dog.*.jpg dog
```
