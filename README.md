# Credit Portfolio Segmentation and Risk Pattern Analysis

## 1. Project Overview

This project is part of my AIML Course from HCL-Guvi. It focuses on analyzing borrower data and grouping borrowers based on their financial and credit characteristics using Machine Learning.

The main idea is to understand different borrower groups, study their financial patterns, and analyze historical loan default rates. I utilized clustering algorithms to identify borrower segments and Isolation Forest to find unusual borrower records. I have also developed an interactive Streamlit dashboard to display the results.

## 2. Project Objectives

- Understand borrower and loan-related data.
- Perform data cleaning and preprocessing.
- Group borrowers based on their financial characteristics.
- Apply different clustering algorithms.
- Visualize borrower groups using dimensionality reduction techniques.
- Compare clustering performance using evaluation metrics.
- Analyze historical default rates across clusters.
- Identify unusual borrower records using anomaly detection.
- Build an interactive dashboard using Streamlit.

## 3. Project Workflow

### Step 1: Data Loading and Understanding
- Loaded the loan dataset using Pandas.
- Checked the number of rows and columns.
- Examined data types and basic statistics.
- Identified numerical and categorical columns.

### Step 2: Data Cleaning and Preprocessing
- Checked missing values and duplicate records.
- Handled incorrect data types.
- Examined outliers.
- Converted date columns into the required format.
- Created additional features from loan issue dates.

### Step 3: Exploratory Data Analysis (EDA)
- Studied numerical and categorical features.
- Created different graphs to understand data distributions.
- Analyzed relationships between financial variables.
- Used correlation analysis to understand feature relationships.

### Step 4: Feature Engineering and Scaling
- Selected important numerical and categorical features.
- Handled missing values using imputation.
- Applied RobustScaler to numerical features.
- Used One-Hot Encoding for categorical features.

### Step 5: Dimensionality Reduction

Used three techniques to visualize borrower data in two dimensions:

- **PCA:** To reduce the number of features while retaining important information.
- **t-SNE:** To visualize similarities between borrowers.
- **UMAP:** To explore patterns and groups in the dataset.

### Step 6: Clustering

Applied the following clustering algorithms:

- K-Means Clustering
- Hierarchical Clustering
- DBSCAN

I used the Silhouette Score and Inertia to compare different K values for K-Means and select a suitable number of clusters.

### Step 7: Borrower Segment Analysis

Analyzed the characteristics of each borrower group.

The analysis includes:
- Median FICO score.
- Median Debt-to-Income (DTI) ratio.
- Historical default rate by cluster.

### Step 8: Anomaly Detection

Used the Isolation Forest algorithm to identify unusual borrower records.

Analyzed the detected anomalies to understand their financial characteristics.

## 5. Streamlit Dashboard

Developed an interactive dashboard to present the project results.

The dashboard contains five sections:

1. **Overview:** Displays important portfolio information and borrower statistics.
2. **Borrower Segmentation:** Shows borrower distribution and cluster characteristics.
3. **Dimensionality Reduction:** Visualizes PCA, t-SNE, and UMAP results.
4. **Clustering Evaluation:** Displays clustering evaluation metrics and K-Means results.
5. **Default & Anomaly Analysis:** Shows historical default rates and detected anomalies.


## 6. Learning Outcomes

Through this project, I gained practical experience in:

- Data cleaning and preprocessing.
- Exploratory Data Analysis.
- Feature engineering and scaling.
- Unsupervised Machine Learning.
- Clustering algorithms and evaluation.
- Dimensionality reduction techniques.
- Anomaly detection.
- Building interactive dashboards using Streamlit.

## 7. Disclaimer

This project is developed for learning and educational purposes. The analysis is intended to understand borrower segments and historical credit patterns. It should not be used as the sole basis for real-world lending or credit approval decisions.

