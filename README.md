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
```
classifier = ERF()
```

Train the model on the training set, with the attrributes *X_train* and the labels *y_train* defined on $2^M$, with *M* the number of classes :
```
classifier.fit(X_train, y_train)
```

Use score to predict the classes of *X_test*, comparegit them to *y_test* and return the accuracy of the model:
```
precisions = classifier.score(X_test, y_test)
```
