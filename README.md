# Churn Prediction for Telecom Customers Using Logistic Regression and XGBoost


## Business Objective
### To uncover the main reasons behind customer churn and develop predictive models that spot at-risk customers early. This allows the business to take proactive steps to retain customers, improve retention strategies, and enhance long-term customer value


## Data Analysis Workflow
### 1. Data Cleaning
- Removed null values in TotalCharges
- Dropped irrelevant columns like customerID
### 2. Exploratory Data Analysis (EDA)
- Performed univariate analysis to explore the distribution of churn rates, customer tenure, and monthly charges, identifying key patterns.
- Used boxplots and crosstabs for bivariate analysis to uncover how churn relates to features such as contract type, tech support, and payment methods.
- Checked the distribution and skewness of numerical variables (e.g., tenure, monthly charges) to identify outliers and inform data preprocessing.
### 3. Feature Engineering
- Log-transformed skewed variables
- Standardised numerical features
- One-hot encoded categorical variables
- Created derived features like number of services (num_service), contract bins
### 4. Modelling
- Built two models: Logistic Regression (interpretable) and XGBoost (performance-oriented)
- Evaluated using **AUC, precision, and recall**, with a focus on high-value customer detection
- Tuned hyperparameters using MLflow for experiment tracking
### 5. Deployment
- Deployed final model using Gradio and Hugging Face Spaces for public demonstration

## Key Findings:
- **Churn Rate**: ~26.5% of customers churned.
- **Monthly Charges**: Churned customers tend to have higher monthly charges.
- **Tenure**: Customers with shorter tenure are more likely to churn.
- **Contract Type**: Month-to-month contracts are more prone to churn than long-term ones.
- **Tech Support**: Lack of tech support correlates with higher churn.

## Business Insights
- **High-Risk Segment**: Customers on month-to-month contracts with high charges and no tech support.
- **Retention Strategy**: Offering discounted long-term contracts or bundled tech support may help reduce churn.
- **Early Warning**: Tenure and monthly charges serve as useful early indicators.
- **Predictive Impact**: Achieved 79% recall, identifying the majority of actual churners. Assuming a 25% retention success rate for targeted interventions, the model could potentially reduce overall churn by approximately 20%.  


## Visualisations

![image](https://github.com/user-attachments/assets/06a2e4b5-0722-4f4a-b494-687e15f1b192)


![image](https://github.com/user-attachments/assets/ae0f0393-37d3-4342-b9cc-eab3569a1804)



![image](https://github.com/user-attachments/assets/3f03892d-668a-4fcd-a38b-5f6567a01e54)


