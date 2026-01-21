import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score  ##for cross validation


## 1.Load the dataset
housing = pd.read_csv("housing.csv")

## 2. Create a Stratified testset
housing['income_cat'] = pd.cut(housing["median_income"],
                               bins = [0.0,1.5,3.0,4.5,6.0,np.inf],
                               labels=[1,2,3,4,5])

split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)

strat_train_set = None ##only done so that interpreter knows variable exists outside loop (not needed but better)
strat_test_set = None
for train_index,test_index in split.split(housing,housing['income_cat']):
    strat_train_set = housing.loc[train_index].drop("income_cat",axis =1) ##we will work on this data
    strat_test_set = housing.loc[test_index].drop("income_cat",axis =1) ##set aside this data


## We will work on set of training data

housing = strat_train_set.copy() ##shows redline bcz if loop doesn't run it is None.copy() which is invalid but loop will run atleast once always so the code is good

## 3.Seperate Predictors and Labels

housing_lables = housing["median_house_value"].copy()
housing = housing.drop("median_house_value",axis =1)

print(housing,housing_lables)

## 4. List the numerical and categorical data
num_attributes = housing.drop("ocean_proximity",axis = 1).columns.tolist() ##stores list of colums in csv without dropped one
cat_attributes = ["ocean_proximity"]

## 5.Now lets make pipeline 
# for numerical attributes
num_pipeline = Pipeline([
    ("imputer",SimpleImputer(strategy = "median")),
    ("scaler",StandardScaler())
])
# for categorical attribute
cat_pipeline = Pipeline([
    ("Encode",OneHotEncoder(handle_unknown = "ignore"))
])

## Contruct a full pipeline
full_pipeline = ColumnTransformer([
    ("num",num_pipeline,num_attributes),
    ("cat",cat_pipeline,cat_attributes)
])

##6. Tranform the data
housing_prepared = full_pipeline.fit_transform(housing)
print(housing_prepared)
print(housing_prepared.shape) ##will give rows and columns in it


#7. Train the model

#Linear Regression model
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared,housing_lables)
lin_preds = lin_reg.predict(housing_prepared)
lin_rmse = root_mean_squared_error(housing_lables,lin_preds)
print(f"The root mean squared for linear regression is {lin_rmse}")
lin_cross = -cross_val_score(lin_reg,housing_prepared,housing_lables,scoring ="neg_root_mean_squared_error",cv = 10)
print(pd.Series(lin_cross).describe()) #will show  10 sets of RMSE values(better to see errors using cross_val_score ie Cross validation)

# Decision Tree  model
dec_reg = DecisionTreeRegressor()
dec_reg.fit(housing_prepared,housing_lables)
dec_preds = dec_reg.predict(housing_prepared)
dec_rmse = root_mean_squared_error(housing_lables,dec_preds)
dec_cross = -cross_val_score(dec_reg,housing_prepared,housing_lables,scoring ="neg_root_mean_squared_error",cv = 10)
print(f"The root mean squared for Decision Tree regression is {dec_rmse}")
print(pd.Series(dec_cross).describe()) ##will show  10 sets of RMSE values(better to see errors using cross_val_score ie Cross validation)

#Random Forest Regressor  model  -> Takes time( but for this set  is giving less error than above two as checked by rmse values with cross validation)
ran_reg = RandomForestRegressor()
ran_reg.fit(housing_prepared,housing_lables)
ran_preds = ran_reg.predict(housing_prepared)
ran_rmse = root_mean_squared_error(housing_lables,ran_preds)
print(f"The root mean squared for Random Forest regression is {ran_rmse}")
ran_cross = -cross_val_score(ran_reg,housing_prepared,housing_lables,scoring ="neg_root_mean_squared_error",cv = 10)
print(pd.Series(ran_cross).describe())
