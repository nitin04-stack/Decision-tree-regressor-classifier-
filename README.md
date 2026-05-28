# Decision Tree - Classification and Regression

This repository contains two notebooks where I explored Decision Trees — one for a classification problem and one for regression.
Both are built using scikit-learn and cover the full workflow from data preparation to model evaluation. 
I kept them in the same repo because the core algorithm is the same, just applied to two different types of problems.

# Notebooks
# 1. Decision_Tree_Classification.ipynb — Titanic Survival Prediction
The goal here is to predict whether a passenger survived the Titanic disaster based on features like age, sex, 
fare, passenger class, and port of embarkation.

#Dataset
Loaded directly from seaborn's built-in datasets (`sns.load_dataset("titanic")`), so no external CSV is needed.

Features used
- sex
- age
- fare
- embarked
- pclass
- Target: survived (0 = Died, 1 = Survived)

What I did
- Handled missing values using `SimpleImputer` — median for age, most frequent for embarked
- Encoded categorical columns (sex, embarked) using `LabelEncoder`
- Trained a base `DecisionTreeClassifier` and evaluated accuracy
- Visualized the tree using `plot_tree`
- Tuned `max_depth` from 2 to 9 to see how depth affects accuracy
- Applied **Cost Complexity Pruning** — used `cost_complexity_pruning_path` to get all possible alpha values, trained a tree for each, and selected the best alpha based on test accuracy
- Visualized the final pruned tree
- 
# 2. Decision_Tree_Regressor.ipynb — Diabetes Disease Progression
This one uses a Decision Tree for regression instead of classification. The task is to predict a quantitative measure of diabetes disease progression based on patient data.

Dataset
Loaded from scikit-learn's built-in datasets (`load_diabetes`), so again no external file needed.

Features used
10 baseline medical variables including age, BMI, blood pressure, and several blood serum measurements.

Target: A continuous value representing disease progression one year after baseline.

What I did
- Confirmed no missing values in the dataset
- Trained a `DecisionTreeRegressor` with `max_depth=5` and `min_samples_leaf=25` to control overfitting
- Evaluated on both train and test sets using **R² Score** and **Mean Squared Error**
- Visualized the full regression tree with `plot_tree`

## Why both are in one repo

Decision Tree is one algorithm that handles both classification and regression problems.
Keeping both notebooks together makes it easy to see how the same underlying logic — splitting data based on feature thresholds
— applies differently depending on whether the output is a category or a continuous number.

# Tech Stack
- Python
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
  
# How to Run
1. Clone the repository
2. Install dependencies:
```bash
pip install pandas matplotlib seaborn scikit-learn
```
3. Open either notebook:
```bash
jupyter notebook decision_tree_classification.ipynb
jupyter notebook Decision_tree_regressor__1_.ipynb
```
No external datasets required — both are loaded directly from seaborn and scikit-learn.

# Repository Structure
```
├── decision_tree_classification.ipynb    # Titanic survival classification
├── Decision_tree_regressor__1_.ipynb     # Diabetes disease regression
└── README.md
```

# What I took away from this

The pruning part in the classification notebook was the most interesting thing to work through. 
Instead of manually guessing a good depth, the CCP alpha approach lets you find the simplest tree that still generalizes well. 
On the regression side, setting `min_samples_leaf` was a good way to prevent the tree from memorizing the training data.
