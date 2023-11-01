# PRODUCT DEMAND PREDICTION WITH MACHINE LEARNING

## Overview:

        Product demand prediction using machine learning is a data-driven approach that leverages historical sales data, product attributes, and other relevant factors to forecast future demand for products. By employing sophisticated machine learning algorithms like XGBoost or LSTM, this predictive model helps businesses optimize inventory management, reduce overstock or stockouts, and enhance customer satisfaction. Accurate predictions enable companies to make informed decisions about production, supply chain, and pricing strategies. This technology-driven solution is a critical tool for modern businesses aiming to streamline operations, enhance resource allocation, and stay competitive in dynamic markets.

## Process Involved In The Project:

        ❖ Data Collection
        ❖ Data Pre-processing
        ❖ Feature Engineering
        ❖ Model Selection
        ❖ Training and Validation 
        ❖ Model Tuning 
        ❖ Prediction and Deployment 
  	    ❖ Monitoring and Updating

## Libraries Used In The Dataset:

        ▪ Numpy
        ▪ Pandas
        ▪ Matplotlib
        ▪ Plotly
       	▪ Scikit-learn

## How To Install The Libraries:
        The Above mentioned Libraries can be downloaded using the following pip command in Command Prompt

        ❖ Numpy         :  pip install numpy
        ❖ Pandas        :  pip install pandas
        ❖ Matplotlib    :  pip install matplotlib
        ❖ Plotly        :  pip install plotly
        ❖ Scikit-Learn  :  pip install Scikit-Learn

## Importing The Packages:

        ## The Following libraries are used to perform the task in the project
        import pandas as pd
        import numpy as np
        import plotly.express as px
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LinearRegression
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        from statsmodels.tsa.arima.model import ARIMA
        from sklearn.metrics import mean_squared_error

## Loading The Dataset:

        ## The read_csv function is used to load the dataset
        data = pd.read_csv("./PoductDemand.csv")

        ## This will return the first five values in the dataset
        data.head()

## Checking the Null Values:

        ## The isnull() Functions is used to find the null values in the dataset and the sum() functions is used to                find the total number of null values in the dataset
        
        data.isnull().sum()

## Making Visualization Using The Following Command:

        ## The Scatter() Functions is used to plot the dataset in the form of visualization
        
        fig = px.scatter(data, x="Units Sold", y="Total Price", size='Units Sold')
        fig.show()

## Spliting The Data As Testing and Training Data:

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## Making The Model To  Predict:

        ## The Fit() Function is used to make Prediction For The Trained data
        
        linear_reg_model.fit(X_train, y_train)
        random_forest_model.fit(X_train, y_train)

## Finding The Linear Regression And  Random Forest:
        For the dataset we have to find the Mean Absolute Error (MAE) and Root Mean Square Error (RMSE) linear                 regression and random forest 

## Making The Predictions By Using a Time Series Algorithm:
        ARIMA - Autoregressive Integrated Moving Average

        ARIMA, or Autoregressive Integrated Moving Average, is a popular time series forecasting model used in statistics and econometrics. It combines autoregressive (AR) and moving average (MA) components with differencing to make a time series data stationary, which simplifies forecasting

        The above mentioned Time Series Forecasting Model is Used to make predictions
