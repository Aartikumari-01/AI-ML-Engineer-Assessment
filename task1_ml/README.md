\# AI4I 2020 Predictive Maintenance



\## 1. Project Overview



This project develops a machine learning-based predictive maintenance system using the AI4I 2020 Predictive Maintenance Dataset.



The objective is to predict machine failures using operational sensor measurements such as air temperature, process temperature, rotational speed, torque, and tool wear.



\## 2. Dataset



\- Dataset: AI4I 2020 Predictive Maintenance Dataset

\- Source: UCI Machine Learning Repository

\- Total Records: 10,000

\- Total Columns: 14

\- Target Variable: Machine failure



Dataset Link: https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset



\## 3. Technologies Used



\- Python

\- Pandas and NumPy

\- Matplotlib and Seaborn

\- Scikit-learn

\- Joblib

\- Jupyter Notebook



\## 4. Project Structure



\- data/: Dataset

\- notebooks/: EDA and model experimentation

\- src/: Preprocessing, training, tuning, feature importance, error analysis, and prediction scripts

\- models/: Saved machine learning models and preprocessor

\- reports/: Evaluation results and analysis

\- reports/figures/: Data visualization graphs



\## 5. Methodology



1\. Exploratory Data Analysis (EDA)

2\. Missing value and duplicate checking

3\. Outlier detection using the IQR method

4\. Feature selection

5\. Stratified train-test split (80:20)

6\. Numerical scaling and categorical encoding

7\. Training Logistic Regression, Random Forest, and SVM

8\. Random Forest hyperparameter tuning

9\. Feature importance analysis

10\. False positive and false negative analysis

11\. Prediction on new machine inputs



\## 6. Model Evaluation



The tuned Random Forest achieved:



\- Accuracy: 98.35%

\- Precision: 90.70%

\- Recall: 57.35%

\- F1 Score: 70.27%

\- ROC-AUC: 96.15%

\- PR-AUC: 80.99%



PR-AUC was considered important because the dataset contains an imbalanced target with only 3.39% machine failures.



\## 7. Installation and Execution



Create and activate a virtual environment:



```bash

python -m venv .venv

.venv\\Scripts\\activate

```



Install the project dependencies using the root requirements.txt file:



```bash

pip install -r requirements.txt

```



Run the scripts from the project root directory:



```bash

python task1\_ml/eda.py

python task1\_ml/eda\_visualization.py

python task1\_ml/src/preprocessing.py

python task1\_ml/src/train\_models.py

python task1\_ml/src/tune\_model.py

python task1\_ml/src/feature\_importance.py

python task1\_ml/src/error\_analysis.py

python task1\_ml/src/predict\_new.py

```



Open the experimentation notebook:



```bash

jupyter notebook task1\_ml/notebooks/eda\_experimentation.ipynb

```



\## 8. Limitations



The dataset is highly imbalanced. Although the tuned model achieves high precision, it misses some actual machine failures. Further recall optimization and validation on real-world machine data are required before deployment.



\## 9. Conclusion



The project demonstrates how machine learning can support predictive maintenance by identifying potential machine failures from sensor data. Random Forest provided the best overall balance among the evaluated baseline models, and hyperparameter tuning further improved its performance.

