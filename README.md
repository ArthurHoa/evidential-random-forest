# Evidential Random Forest

A python version of an Evidential Random Forest based on Evidential Decision Trees.

### Summary

When the data are labeled in an uncertain and imprecise way, Evidential Random Forest model can be used for classification problems.
It is a low variance model improving the performance of Evidential Decision Trees.
The model creates a forest of evidential decision trees estimators using bagging and gives a prediction according to the N estimators is the forest.

### Reference

When using this code please cite and refer to [Paper being published](https://github.com/ArthurHoa/evidential-random-forest)

### How to use

Initialize the model:

```python
classifier = ERF()
```

Train the model on the training set, with the attrributes *X_train* and the labels *y_train* defined on $2^M$, with *M* the number of classes :

```python
classifier.fit(X_train, y_train)
```

Use score to predict the classes of *X_test*, comparegit them to *y_test* and return the accuracy of the model:

```python
precisions = classifier.score(X_test, y_test)
```

## Example on Credal Dog-2

### Credal Dog-2

Link to the dataset : [Credal Dog-2](https://github.com/ArthurHoa/credal-datasets)

Brittany | Beagle
:--:|:--:
<img src="https://github.com/ArthurHoa/credal-datasets/blob/master/ressources/pictures/Brittany.jpg?raw=true" width="70"> |  <img src="https://github.com/ArthurHoa/credal-datasets/blob/master/ressources/pictures/Beagle.jpg?raw=true" width="70">  

### Code

```python
from sklearn.model_selection import train_test_split
from evidential_random_forest import ERF
import numpy as np

X = np.loadtxt('X.csv', delimiter=';')
y = np.loadtxt('y.csv', delimiter=';')
y_true = np.loadtxt('y_true.csv', delimiter=';')

indexes = [i for i in range(X.shape[0])]
train, test, _, _ = train_test_split(indexes, indexes, test_size=.2)

classifier = ERF()

classifier.fit(X[train], y[train])

precision = classifier.score(X[test], y_true[test])

print("Accuracy : ", precision)
```

Accuracy = 0.95

### Output of the model

An exmaple of the Evidential Random Forest prediction for the following picture is given as follows:

<img src="https://www.dropbox.com/s/2wb8u7y7ol95dar/58.jpg?raw=true" width="150">  
  
Prediction:  
m({Brittany}) = 0.27  
m({Beagle}) = 0.47  
m({Beagle, Brittany}) = 0.26  
  
True class: Beagle
