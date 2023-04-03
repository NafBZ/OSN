# Online Social Network Behavioral Analysis

> This repository contains the experimental copy of the paper [SPY-BOT: Machine learning-enabled post filtering for Social Network-Integrated Industrial Internet of Things](https://www.sciencedirect.com/science/article/abs/pii/S1570870521001256)

**Datasets :**

1. [train data](/DataSet/train.csv): Created from a survey
2. [test data](/DataSet/test.csv): Synthetic Data

**Class distribution:**

<img src = "Jpeg/classdist.png" width = "400">


**Algorithm used:** 

1. Logistic Regression
2. Support Vector Machine

**Evaluation Metrices**

1. Accuracy:  approximately 94% on dev set & 90.1% in unseen test set

<img src = "Jpeg/accuracy.png" width = "500">

2. ROC Curve

<img src = "Jpeg/roc.png" width = "500">


## How to run the code

First clone the repository

```
$ git clone https://github.com/NafBZ/OSN.git
```

Then go to the Scripts directory and run the main file for traiining.

```
$ cd Scripts/
$ pytthon3 main.py
```

For testing run the following command.

```
$ pytthon3 test.py
```

To play with the parameters, change the [config.yaml](/config.yaml) file.
