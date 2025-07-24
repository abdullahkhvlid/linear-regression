#Linear Regression Model – Predictive Analysis Project

This repository contains a complete implementation of a Linear Regression model using Python’s Scikit-learn library. The project focuses on building, training, and evaluating a regression model on structured tabular data. It includes error analysis, residual diagnostics, and proper validation using industry-standard metrics.

Objectives

Implement and train a Linear Regression model

Evaluate prediction accuracy using R² score and Mean Squared Error (MSE)

Visualize residual distributions to assess model fit

Interpret statistical performance and limitations

Understand regression error behavior and impact of data quality

Dataset

A clean, structured dataset was used consisting of continuous numeric features and a target variable suitable for regression modeling. The dataset was split into training and testing sets using an 80/20 ratio.

Technologies Used

Python 3.10+

pandas

numpy

matplotlib

seaborn

scikit-learn

Core Workflow

Data Preprocessing

Handled missing values (if present)

Split data into independent variables X and dependent variable y

Applied train_test_split to ensure unbiased evaluation

Model Training

python
Copy
Edit
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, y_train)
The model learns coefficients (slopes) to minimize the least squares loss function.

Predictions and Evaluation

python
Copy
Edit
predictions = reg.predict(X_test)
Evaluated performance using:

R² Score (r2_score): Indicates the proportion of variance explained by the model.

Mean Squared Error (MSE): Measures average squared difference between predicted and actual values.

python
Copy
Edit
from sklearn.metrics import r2_score, mean_squared_error

r2 = r2_score(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
Example Output:

R² Score: 0.33 → Approx. 33% of variance captured

MSE: e.g. 2421.1

Error Distribution Analysis

python
Copy
Edit
import seaborn as sns
sns.displot(y_test - predictions)
The residual plot helps diagnose skewness, outliers, and how well the model generalized.

Interpretation

A low R² score suggests the data may not have a strong linear correlation or may require feature engineering or advanced models.

The residual distribution indicates prediction errors and the presence of potential outliers.

Limitations

Linear Regression assumes linearity, homoscedasticity, and independence between predictors.

It is sensitive to outliers and multicollinearity.

May underperform if features are not scaled or if important nonlinear relationships exist.

License

This project is released under the MIT License.
