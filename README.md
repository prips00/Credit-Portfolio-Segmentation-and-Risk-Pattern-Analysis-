# Project-1
This project is part of my AIML course curriculum from HCL-Guvi. I have done Credit Portfolio Segmentation and  Risk Patten Analysis using Machine Learning.

## Credit Portfolio Segmentation and Risk Pattern Analysis

1. Project Overview
Credit Portfolio Segmentation and Risk Pattern Analysis is a Machine Learning project that aims to identify groups of borrowers with similar financial and credit characteristics using unsupervised learning techniques.
The project applies clustering algorithms to segment borrowers based on features such as revenue, debt-to-income ratio, loan amount, FICO score and employment experience.

2. Project Objectives
Perform data quality analysis and preprocessing.
Explore financial and credit-related characteristics of borrowers.
Identify potential outliers in numerical features.
Apply K-Means clustering to segment borrowers.
Evaluate clustering quality using suitable evaluation metrics.
Profile clusters based on financial and credit characteristics.
Compare historical default rates across identified clusters.
Generate business insights from borrower segmentation.

3. Dataset Description
Dataset: loans_granting_model_dataset.csv
Dataset size: 1,347,681 records
Number of columns: 15
The dataset contains borrower-related financial, employment, geographical and loan information.
Key Features
Feature
Description
revenue
Borrower's revenue
dti_n
Debt-to-income ratio
loan_amnt
Loan amount
fico_n
FICO credit score
experience_c
Employment experience-related feature
emp_length
Employment length
purpose
Loan purpose
home_ownership_n
Home ownership status
addr_state
Borrower's state
issue_d
Loan issue date
Default
Historical default indicator

Note: The Default variable is excluded from clustering and used only for post-clustering historical default analysis.

4. Technologies and Libraries
Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
Jupyter Notebook

5. Project Workflow
Step 1: Data Loading and Understanding
Load the dataset using Pandas.
Examine dataset dimensions, data types and statistical summaries.
Understand numerical and categorical features.
Step 2: Data Quality Analysis
Identify missing values.
Check duplicate records and duplicate IDs.
Examine unique values in categorical columns.
Identify constant features.
Check inconsistent values and data types.
Step 3: Data Cleaning and Preprocessing
Handle missing values.
Remove duplicate records where necessary.
Standardize categorical text values.
Convert date columns into appropriate formats.
Prepare numerical and categorical features for analysis.
Step 4: Exploratory Data Analysis
Univariate analysis.
Numerical feature distributions.
Categorical feature distributions.
Boxplot analysis.
Potential outlier detection using the IQR method.
Step 5: Feature Preparation
Select relevant financial and credit features.
Apply numerical imputation and scaling.
Encode categorical variables.
Prepare the feature matrix for clustering.
Step 6: Model Development
Apply MiniBatch K-Means clustering.
Select the number of clusters using model evaluation.
Train the final clustering model.
Assign cluster labels to borrowers.
Step 7: Clustering Evaluation
Evaluate clustering quality using:
Silhouette Score
Davies-Bouldin Index
Calinski-Harabasz Index
Step 8: Cluster Profiling
Analyse each cluster based on:
Average and median revenue.
Debt-to-income ratio.
Loan amount.
FICO score.
Employment experience.
Loan purpose.
Home ownership.
Step 9: Credit Risk Analysis
Compare historical default rates across clusters.
Identify borrower segments with different historical default patterns.
Examine geographical and temporal patterns where relevant.


