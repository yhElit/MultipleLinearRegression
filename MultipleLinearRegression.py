# Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn import metrics


def main():
    # Reading the dataset
    dataset = pd.read_csv("data.csv")
    dataset.head()

    # Data Pre-Processing
    dataset.shape

    # Checking for missing values
    dataset.isna().sum()

    # Checking for duplicate rows
    dataset.duplicated().any()

    # Checking for outliers
    fig, axs = plt.subplots(3, figsize=(5, 5))
    plt1 = sns.boxplot(dataset['TV'], ax=axs[0])
    plt2 = sns.boxplot(dataset['Newspaper'], ax=axs[1])
    plt3 = sns.boxplot(dataset['Radio'], ax=axs[2])
    plt.tight_layout()

    # Distribution of the target variable
    sns.distplot(dataset['Sales']);

    # How Sales are related with other variables
    sns.pairplot(dataset, x_vars=['TV', 'Radio', 'Newspaper'], y_vars='Sales', height=4, aspect=1, kind='scatter')
    plt.show()

    # Heatmap
    sns.heatmap(dataset.corr(), annot=True)
    plt.show()

    # Multiple Regression
    # Setting the value for X and Y
    x = dataset[['TV', 'Radio', 'Newspaper']]
    y = dataset['Sales']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=100)
    mlr = LinearRegression()
    mlr.fit(x_train, y_train)
    LinearRegression()

    # Printing the model coefficients
    print(mlr.intercept_)
    # pair the feature names with the coefficients
    list(zip(x, mlr.coef_))

    # Predicting the Test and Train set result
    y_pred_mlr = mlr.predict(x_test)
    x_pred_mlr = mlr.predict(x_train)

    print("Prediction for test set: {}".format(y_pred_mlr))

    # Actual value and the predicted value
    mlr_diff = pd.DataFrame({'Actual value': y_test, 'Predicted value': y_pred_mlr})
    mlr_diff

    # Predict for any value
    mlr.predict([[56, 55, 67]])

    # print the R-squared value for the model
    print('R squared value of the model: {:.2f}'.format(mlr.score(x, y) * 100))

    # 0 means the model is perfect. Therefore the value should be as close to 0 as possible
    meanAbErr = metrics.mean_absolute_error(y_test, y_pred_mlr)
    meanSqErr = metrics.mean_squared_error(y_test, y_pred_mlr)
    rootMeanSqErr = np.sqrt(metrics.mean_squared_error(y_test, y_pred_mlr))

    print('Mean Absolute Error:', meanAbErr)
    print('Mean Square Error:', meanSqErr)
    print('Root Mean Square Error:', rootMeanSqErr)


if __name__ == '__main__':
    main()
