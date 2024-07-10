

# House Price Prediction

The project involves creating the best model for predicting house prices based on the "Poland OLX House Price Q122" dataset. 
https://www.kaggle.com/datasets/g1llar/poland-olx-house-price-q122




# Project description
The project includes an 'EDA' section dedicated to analyzing the dataset. This involves, among other things: examining the distribution of numerical variables, checking the impact of categorical variables on the price, and assessing the impact of numerical variables on the price (e.g. location of the house (longitude, latitude), house size, number of rooms, and floors). Then, the dataset is cleaned of outliers for individual columns. 

The next section includes data normalization (MinMax Scaler) and splitting the data into train/dev/test sets.

# Models

To achieve the best results, I used three different machine learning algorithms 

- Linear Regression 
- DecisionTree
- Neural Network



# Results for models

### Linear Regression Results : 

![LinearRegressionResults](./readmeimages/lin_ress.png)

### Decision Tree Results : 

![DecisionTreeResults](./readmeimages/tree_res.png)

### Neural Network Results : 

![NNResults](./readmeimages/nn_res.png)

![NNLossResults](./readmeimages/nnlos_res.png)

# Summary

Among the three models used, linear regression performed the worst with an accuracy of only 65%. The Neural Network and Decision Tree models excelled at this task. Both the Decision Tree and Neural Network achieved an accuracy of 99%. When plotting y_test against y_pred, a linear (1:1) relationship appears, similarly for the Neural Network.