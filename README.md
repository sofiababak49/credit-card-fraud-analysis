# Credit Card Fraud Analysis
## Objective
Credit card fraud is a common issue for banks and their customers.  In this project, I analyse a public credit card transaction dataset using Python. The main goal is to understand what differences can be found between fraudulent and normal transactions, and to test how well two basic machine learning models can detect fraud.
The analysis focuses on four main questions:
1. How common is fraud in the dataset?
2. Do fraudulent transactions have different amounts?
3. Are there any patterns in transaction activity over time?
4. Do unusually large transactions have a higher fraud rate?
## Dataset
The project uses [the Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) dataset from Kaggle. It contains 284,807 transactions made by European cardholders over two days in September 2013. Only 492 transactions are labelled as fraudulent. The dataset has 31 columns: 
- Time: seconds passed since the first transaction
- Amount: transaction amount
- Class: 0 = normal transaction, 1 = fraud
- V1- V28`: anonymised features ocreated using PCA (their original meaning is not available)
The dataset file is not included in this repository because of its size. To run the notebook, download creditcard.csv from Kaggle and place it in the same folder as the notebook.
## Tools
Python, Pandas, NumPy, Matplotlib, Scikit-learn, Imbalanced-learn, Jupyter Notebook
## Analysis 
The project includes the following steps:
- Data quality check: checking missing values and duplicates (1,081 duplicate rows were removed, leaving 283,726 transactions)
- Class imbalance: checking the fraud percentage and why accuracy is misleading
- Transaction amounts: fraud vs normal, looking at their distributions and checking unusually large transactions
- Activity over time: analysing the number of transactions and fraud rates for different hours
- Correlation analysis: checking which features have the strongest relationships with fraud
- Models: Logistic Regression and Random Forest. Both models were trained on an undersampled training set and tested on an untouched test set. The evaluation includes recall, precision, F1-score, ROC-AUC and PR-AUC. 
## Key Findings
1. Fraud is very rare in this dataset. After removing duplicates, there are 473 fraudulent transactions out of 283,726 (0.167%). This also shows why accuracy can be misleading. A model that predicts every transaction as normal would still have 99.83% accuracy.
2. Fraudulent transactions have a lower median amount (9.82 vs 22.00), but the amounts overlap quite a lot, so amount alone is not enough to identify fraud.
3. Among the largest 1% of transactions (above 1,018.96), the fraud rate is 0.317% compared to 0.167% across the whole dataset. Still, only 9 transactions in this group are fraudulent. Most fraud cases therefore involve smaller amounts.
4. The highest fraud rates appear at hours 26, 28 and 2 (1.34-1.56%). These rates are around 8-9 times higher than the overall rate. Meanwhile, the highest number of fraud cases (43) occurs at hour 11. 
5. V17, V14 and V12 have the strongest correlations with fraud. However, these relationships are only moderate, so correlation alone does not explain fraud patterns.
6. Both models detect around 80% of fraud cases. Random Forest has better precision (0.53 vs 0.30) and PR-AUC (0.689 vs 0.652). It also produces fewer false alarms (66 vs 184). 
## Limitations
- V1-V28 are anonymised, so their exact meaning cannot be interpreted.
- The dataset covers only 2 days, and Time is measured from the first transaction, not the actual time of day.
- The test set contains only 95 fraudulent transactions, which makes the evaluation metrics less reliable.
- Undersampling removed most normal transactions from the training set. As a result, the models had less information about normal transactions during training.
- I used only one train-test split and did not perform hyperparameter tuning.
