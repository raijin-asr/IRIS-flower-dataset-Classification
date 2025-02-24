# IRIS-flower-dataset-Classification

This repository contains code for classifying Iris flowers into their respective species (setosa, versicolor, or virginica) using machine learning.  The project utilizes the well-known Iris dataset and explores several classification algorithms.

## Dataset

The [Iris dataset](https://archive.ics.uci.edu/ml/datasets/iris) is used for training and evaluating the models. It contains 150 samples of Iris flowers, with four features measured for each sample:

* Sepal Length (cm)
* Sepal Width (cm)
* Petal Length (cm)
* Petal Width (cm)

Each sample is labeled with one of three species:

* Setosa
* Versicolor
* Virginica

The dataset is available as `IRIS.csv` in the repository.

## Project Overview

The code in this repository performs the following steps:

1. **Data Loading and Preprocessing:** The Iris dataset is loaded using pandas. The 'Iris-' prefix is removed from the species names for simplicity. The data is then split into features (X) and the target variable (y).  A train-test split is performed to evaluate the model's performance on unseen data.  A smaller test size of 20 samples is used as per the original notebook.  **Note:** While the original notebook didn't include feature scaling, this is often beneficial for algorithms like KNN, SVM, and Logistic Regression.  A best practice would be to add a `StandardScaler` to a `ColumnTransformer` and `Pipeline` for preprocessing.  *(See suggested improvement below)*

2. **Model Training and Evaluation:** Several classification algorithms are used:
    * K-Nearest Neighbors (KNN)
    * Random Forest Classifier
    * Support Vector Machine (SVM)
    * Logistic Regression

   Cross-validation (4-fold) is used to estimate the performance of each model on the training data. The mean accuracy and standard deviation are calculated for each model.

3. **Model Selection and Final Training:** The best performing model (based on cross-validation accuracy) is selected. This best model is then trained on the *entire* training dataset.

4. **Test Set Evaluation:** The trained best model is evaluated on the held-out test set to get an estimate of its real-world performance.  The test accuracy is printed.

5. **Submission File Generation:** The predictions on the test set are saved to a CSV file named `Datalligence_submission.csv` in the format required for potential submission (with an ID column).

## Running the Code

To run the code, you will need to have the following libraries installed:

* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn

You can install these libraries using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
