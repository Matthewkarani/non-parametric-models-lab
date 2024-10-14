# Nonparametric ML Models - Cumulative Lab

## Introduction

In this cumulative lab, you will apply two nonparametric models you have just learned — k-nearest neighbors and decision trees — to the forest cover dataset.

## Objectives

* Practice identifying and applying appropriate preprocessing steps
* Perform an iterative modeling process, starting from a baseline model
* Explore multiple model algorithms, and tune their hyperparameters
* Practice choosing a final model across multiple model algorithms and evaluating its performance

## Your Task: Complete an End-to-End ML Process with Nonparametric Models on the Forest Cover Dataset

![line of pine trees](https://curriculum-content.s3.amazonaws.com/data-science/images/trees.jpg)

Photo by <a href="https://unsplash.com/@michaelbenz?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Michael Benz</a> on <a href="/s/photos/forest?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>

### Business and Data Understanding

To repeat the previous description:

> Here we will be using an adapted version of the forest cover dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/covertype). Each record represents a 30 x 30 meter cell of land within Roosevelt National Forest in northern Colorado, which has been labeled as `Cover_Type` 1 for "Cottonwood/Willow" and `Cover_Type` 0 for "Ponderosa Pine". (The original dataset contained 7 cover types but we have simplified it.)

The task is to predict the `Cover_Type` based on the available cartographic variables:


```python
# Run this cell without changes
import pandas as pd

df = pd.read_csv('data/forest_cover.csv')
df
```

> As you can see, we have over 38,000 rows, each with 52 feature columns and 1 target column:

> * `Elevation`: Elevation in meters
> * `Aspect`: Aspect in degrees azimuth
> * `Slope`: Slope in degrees
> * `Horizontal_Distance_To_Hydrology`: Horizontal dist to nearest surface water features in meters
> * `Vertical_Distance_To_Hydrology`: Vertical dist to nearest surface water features in meters
> * `Horizontal_Distance_To_Roadways`: Horizontal dist to nearest roadway in meters
> * `Hillshade_9am`: Hillshade index at 9am, summer solstice
> * `Hillshade_Noon`: Hillshade index at noon, summer solstice
> * `Hillshade_3pm`: Hillshade index at 3pm, summer solstice
> * `Horizontal_Distance_To_Fire_Points`: Horizontal dist to nearest wildfire ignition points, meters
> * `Wilderness_Area_x`: Wilderness area designation (3 columns)
> * `Soil_Type_x`: Soil Type designation (39 columns)
> * `Cover_Type`: 1 for cottonwood/willow, 0 for ponderosa pine

This is also an imbalanced dataset, since cottonwood/willow trees are relatively rare in this forest:


```python
# Run this cell without changes
print("Raw Counts")
print(df["Cover_Type"].value_counts())
print()
print("Percentages")
print(df["Cover_Type"].value_counts(normalize=True))
```

Thus, a baseline model that always chose the majority class would have an accuracy of over 92%. Therefore we will want to report additional metrics at the end.

### Previous Best Model

In a previous lab, we used SMOTE to create additional synthetic data, then tuned the hyperparameters of a logistic regression model to get the following final model metrics:

* **Log loss:** 0.13031294393913376
* **Accuracy:** 0.9456679825472678
* **Precision:** 0.6659919028340081
* **Recall:** 0.47889374090247455

In this lab, you will try to beat those scores using more-complex, nonparametric models.

### Modeling

Although you may be aware of some additional model algorithms available from scikit-learn, for this lab you will be focusing on two of them: k-nearest neighbors and decision trees. Here are some reminders about these models:

#### kNN - [documentation here](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)

This algorithm — unlike linear models or tree-based models — does not emphasize learning the relationship between the features and the target. Instead, for a given test record, it finds the most similar records in the training set and returns an average of their target values.

* **Training speed:** Fast. In theory it's just saving the training data for later, although the scikit-learn implementation has some additional logic "under the hood" to make prediction faster.
* **Prediction speed:** Very slow. The model has to look at every record in the training set to find the k closest to the new record.
* **Requires scaling:** Yes. The algorithm to find the nearest records is distance-based, so it matters that distances are all on the same scale.
* **Key hyperparameters:** `n_neighbors` (how many nearest neighbors to find; too few neighbors leads to overfitting, too many leads to underfitting), `p` and `metric` (what kind of distance to use in defining "nearest" neighbors)

#### Decision Trees - [documentation here](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)

Similar to linear models (and unlike kNN), this algorithm emphasizes learning the relationship between the features and the target. However, unlike a linear model that tries to find linear relationships between each of the features and the target, decision trees look for ways to split the data based on features to decrease the entropy of the target in each split.

* **Training speed:** Slow. The model is considering splits based on as many as all of the available features, and it can split on the same feature multiple times. This requires exponential computational time that increases based on the number of columns as well as the number of rows.
* **Prediction speed:** Medium fast. Producing a prediction with a decision tree means applying several conditional statements, which is slower than something like logistic regression but faster than kNN.
* **Requires scaling:** No. This model is not distance-based. You also can use a `LabelEncoder` rather than `OneHotEncoder` for categorical data, since this algorithm doesn't necessarily assume that the distance between `1` and `2` is the same as the distance between `2` and `3`.
* **Key hyperparameters:** Many features relating to "pruning" the tree. By default they are set so the tree can overfit, and by setting them higher or lower (depending on the hyperparameter) you can reduce overfitting, but too much will lead to underfitting. These are: `max_depth`, `min_samples_split`, `min_samples_leaf`, `min_weight_fraction_leaf`, `max_features`, `max_leaf_nodes`, and `min_impurity_decrease`. You can also try changing the `criterion` to "entropy" or the `splitter` to "random" if you want to change the splitting logic.

### Requirements

#### 1. Prepare the Data for Modeling

#### 2. Build a Baseline kNN Model

#### 3. Build Iterative Models to Find the Best kNN Model

#### 4. Build a Baseline Decision Tree Model

#### 5. Build Iterative Models to Find the Best Decision Tree Model

#### 6. Choose and Evaluate an Overall Best Model

## 1. Prepare the Data for Modeling

The target is `Cover_Type`. In the cell below, split `df` into `X` and `y`, then perform a train-test split with `random_state=42` and `stratify=y` to create variables with the standard `X_train`, `X_test`, `y_train`, `y_test` names.

Include the relevant imports as you go.


```python
# Your code here

from sklearn.model_selection import train_test_split

X = df.drop("Cover_Type", axis=1)
y = df["Cover_Type"]


X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=42, stratify=y )
```

Now, instantiate a `StandardScaler`, fit it on `X_train`, and create new variables `X_train_scaled` and `X_test_scaled` containing values transformed with the scaler.


```python
# Your code here

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

The following code checks that everything is set up correctly:


```python
# Run this cell without changes

# Checking that df was separated into correct X and y
assert type(X) == pd.DataFrame and X.shape == (38501, 52)
assert type(y) == pd.Series and y.shape == (38501,)

# Checking the train-test split
assert type(X_train) == pd.DataFrame and X_train.shape == (28875, 52)
assert type(X_test) == pd.DataFrame and X_test.shape == (9626, 52)
assert type(y_train) == pd.Series and y_train.shape == (28875,)
assert type(y_test) == pd.Series and y_test.shape == (9626,)

# Checking the scaling
assert X_train_scaled.shape == X_train.shape
assert round(X_train_scaled[0][0], 3) == -0.636
assert X_test_scaled.shape == X_test.shape
assert round(X_test_scaled[0][0], 3) == -1.370
```

## 2. Build a Baseline kNN Model

Build a scikit-learn kNN model with default hyperparameters. Then use `cross_val_score` with `scoring="neg_log_loss"` to find the mean log loss for this model (passing in `X_train_scaled` and `y_train` to `cross_val_score`). You'll need to find the mean of the cross-validated scores, and negate the value (either put a `-` at the beginning or multiply by `-1`) so that your answer is a log loss rather than a negative log loss.

Call the resulting score `knn_baseline_log_loss`.

Your code might take a minute or more to run.


```python
# Replace None with appropriate code

# Relevant imports
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import numpy as np


# Creating the model
knn_baseline_model = KNeighborsClassifier()

# Perform cross-validation
knn_baseline_log_loss = cross_val_score(knn_baseline_model , X_train_scaled, y_train, scoring="neg_log_loss")

knn_baseline_log_loss = -np.mean(neg_log_loss_scores)

knn_baseline_log_loss
```
#### Outcome: 0.12964546386734574


Our best logistic regression model had a log loss of 0.13031294393913376

Is this model better? Compare it in terms of metrics and speed.


```python
# Replace None with appropriate text
"""
Yes the model is better, is has a better log loss. 

"""
```

## 3. Build Iterative Models to Find the Best kNN Model

Build and evaluate at least two more kNN models to find the best one. Explain why you are changing the hyperparameters you are changing as you go. These models will be *slow* to run, so be thinking about what you might try next as you run them.


#### In the model below, we will increase the neighbors hyperparameter, this will smooth the decision boundaries which may help reduce variance and improve generalization on the test data.


```python
# Your code here (add more cells as needed)

# Model 1: Increase n_neighbors to 10
knn_model_1 = KNeighborsClassifier(n_neighbors=10)

# Perform cross-validation with the new model
neg_log_loss_scores_1 = cross_val_score(knn_model_1, X_train_scaled, y_train, scoring="neg_log_loss", cv=5)

# Calculate the log loss for model 1
knn_model_1_log_loss = -np.mean(neg_log_loss_scores_1)

knn_model_1_log_loss
```
##### Outcome: 0.07502202520388171

##### In the model below, we will change the weights to 'distance'.. this allows closer neighbors to have influence than further ones. This might improve performance in cases where nearby points are more reliable for predictions. 


```python
# Your code here (add more cells as needed)
# Model 2: Change the weights to 'distance' with n_neighbors=10
knn_model_2 = KNeighborsClassifier(n_neighbors=10, weights='distance')

# Performs cross validation with the new model. 
neg_log_loss_scores_2 = cross_val_score(knn_model_2, X_train_scaled,y_train,scoring="neg_log_loss", cv=5)

# Calculates the log loss for model 2
knn_model_2_log_loss = -np.mean(neg_log_loss_scores_2)

knn_model_2_log_loss
```
##### Outcome: 0.06982850921022257


##### In the model below, we will change the distance metric hyperparameter. The default distance hyperparameter is Minkowski distance(p=2) which corresponds to Euclidean distance. We will switch it to p=1.

```python
# Your code here (add more cells as needed)
# Model 3: Change distance metric to Manhattan (p=1)
knn_model_3 = KNeighborsClassifier(n_neighbors=10, weights='distance', p=1)

# Perform cross-validation with the new model
neg_log_loss_scores_3 = cross_val_score(knn_model_3, X_train_scaled, y_train, scoring="neg_log_loss", cv=5)

# Calculate the log loss for model 3
knn_model_3_log_loss = -np.mean(neg_log_loss_scores_3)

knn_model_3_log_loss

```
##### Outcome: 0.05886194676088591

## 4. Build a Baseline Decision Tree Model

Now that you have chosen your best kNN model, start investigating decision tree models. First, build and evaluate a baseline decision tree model, using default hyperparameters (with the exception of `random_state=42` for reproducibility).

(Use cross-validated log loss, just like with the previous models.)


```python
# Your code here
from sklearn.tree import DecisionTreeClassifier


# Initialize baseline Decision Tree Model with random state of 42
baseline_decision_tree = DecisionTreeClassifier(random_state=42)

# Perform cross validation 
neg_log_loss_scores_dt = cross_val_score(baseline_decision_tree, X_train_scaled, y_train, scoring="neg_log_loss", cv=5)

# Calculate the average log loss (convert negative log loss to positive)
baseline_decision_tree_log_loss = -np.mean(neg_log_loss_scores_dt)

baseline_decision_tree_log_loss
```

##### Outcome: 0.7364763809378052

Interpret this score. How does this compare to the log loss from our best logistic regression and best kNN models? Any guesses about why?


```python
# Replace None with appropriate text
"""
It performs better than our best logistic regression model and our best knn models but it is most likely overfitting(it has learnt the training data very well and is unable
to generalize) this is most likely because decision trees are prone to overfitting.

"""
```

## 5. Build Iterative Models to Find the Best Decision Tree Model

Build and evaluate at least two more decision tree models to find the best one. Explain why you are changing the hyperparameters you are changing as you go.


##### In this first model , we will restrict the maximum tree depth this is because a deep decision tree is prone to overfitting. By limiting the depth of the tree with max_depth, we can prevent the model from learning overly complex patterns which improves generalization. 


```python
# Your code here (add more cells as needed)
# Initialize the model (Limit the depth of the tree to 5)
decision_tree_model_1 = DecisionTreeClassifier(max_depth=5, random_state=42)

# perform cross-validation using log loss
neg_log_loss_scores_dt_1 = cross_val_score(decision_tree_model_1, X_train_scaled, y_train, scoring="neg_log_loss", cv=5)

# Calculate the log loss for the model 
decision_tree_model_1_log_loss = -np.mean(neg_log_loss_scores_dt_1)

decision_tree_model_1_log_loss
```
#### Outcome  : 0.12054652328854479

##### In the model below , we will control the Minimum Samples per Split and Leaf..This prevents the tree from splitting nodes based on very small samples. Which can lead to high variance and overfitting.

```python
# Your code here (add more cells as needed)
# Initialize the model with changed hyperparameters
decision_tree_model_2 = DecisionTreeClassifier(max_depth=5, 
                                               min_samples_split=10, 
                                               min_samples_leaf=5,
                                                 random_state=42)


# Perform cross-validation using log loss
neg_log_loss_scores_dt_2 = cross_val_score(decision_tree_model_2, X_train_scaled, y_train, scoring="neg_log_loss", cv=5)


# Calculate the log los for model 2
decision_tree_model_2_log_loss = -np.mean(neg_log_loss_scores_dt_2)

decision_tree_model_2_log_loss
```
#### Outcome : 0.11733519649174132

#### In the model below , we will further tune the Model Tree Depth and Minimum Samples, this is due to the fact that there was an improvement in the model 2 from model 1. Our aim is to get the right balance between bias and variance.

```python
# Your code here (add more cells as needed)

# Initialize the decision tree model
decision_tree_model_3 = DecisionTreeClassifier(max_depth=5,
                                                min_samples_split=12,
                                                  min_samples_leaf=10,
                                                    random_state=42)

# Perform cross-validation using log loss
neg_log_loss_scores_dt_3 = cross_val_score(decision_tree_model_3, X_train_scaled, y_train, scoring="neg_log_loss", cv=5)

# Calculate the log loss for model 3
decision_tree_model_3_log_loss = -np.mean(neg_log_loss_scores_dt_3)

decision_tree_model_3_log_loss
```
#### Outcome : 0.11638171592158686

##### In the above final decision tree, we increased the min_samples_split and we got a better log loss score even if by only one unit compared to the previous model.

## 6. Choose and Evaluate an Overall Best Model

Which model had the best performance? What type of model was it?

Instantiate a variable `final_model` using your best model with the best hyperparameters.


```python
# Replace None with appropriate code
final_model = KNeighborsClassifier(n_neighbors=10, weights='distance')

# Fit the model on the full training data
# (scaled or unscaled depending on the model)
final_model.fit(X_train_scaled,y_train)
```

Now, evaluate the log loss, accuracy, precision, and recall. This code is mostly filled in for you, but you need to replace `None` with either `X_test` or `X_test_scaled` depending on the model you chose.


```python
# Replace None with appropriate code
from sklearn.metrics import accuracy_score, precision_score, recall_score,log_loss

preds = final_model.predict(X_test_scaled)
probs = final_model.predict_proba(X_test_scaled)

print("log loss: ", log_loss(y_test, probs))
print("accuracy: ", accuracy_score(y_test, preds))
print("precision:", precision_score(y_test, preds))
print("recall:   ", recall_score(y_test, preds))
```

#### Output: 
log loss:  0.08141386112087046
accuracy:  0.9822356118844795
precision: 0.910828025477707
recall:    0.8326055312954876

Interpret your model performance. How would it perform on different kinds of tasks? How much better is it than a "dummy" model that always chooses the majority class, or the logistic regression described at the start of the lab?


```python
# Replace None with appropriate text
"""
Due to the fact that this model had class imbalance there is a high 
probability that the model is choosing the majority class when making the 
predictions. To fix this, I will apply resampling techniques, use class weights, or consider ensemble methods to better balance the classes and improve model 
performance.

"""
```

## Conclusion

In this lab, you practiced the end-to-end machine learning process with multiple model algorithms, including tuning the hyperparameters for those different algorithms. You saw how nonparametric models can be more flexible than linear models, potentially leading to overfitting but also potentially reducing underfitting by being able to learn non-linear relationships between variables. You also likely saw how there can be a tradeoff between speed and performance, with good metrics correlating with slow speeds.
