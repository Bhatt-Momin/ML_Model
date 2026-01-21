import os
import joblib
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
from sklearn.model_selection import cross_val_score

MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"  ##these two files will be created by joblib as soon data is dumped in them (see forward code)

def build_pipeline(nums_attributes,cat_attributes):
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

    return full_pipeline

if not os.path.exists(MODEL_FILE):
    #lets train the model
    housing = pd.read_csv("housing.csv")

    ## 2. Create a Stratified testset
    housing['income_cat'] = pd.cut(housing["median_income"],
                                bins = [0.0,1.5,3.0,4.5,6.0,np.inf],
                                labels=[1,2,3,4,5])

    split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)

    strat_train_set = None ##only done so that interpreter knows variable exists outside loop (not needed but better)
    strat_test_set = None
    for train_index,test_index in split.split(housing,housing['income_cat']):
        housing.loc[test_index].drop("income_cat",axis =1).to_csv("input.csv",index = False) ##all test dataset to transfer in input_csv(created at same time)
        housing =  housing.loc[train_index].drop("income_cat",axis =1) ##we will work on this data
    
    housing_lables = housing["median_house_value"].copy()
    housing_features= housing.drop("median_house_value",axis =1)

    num_attributes = housing_features.drop("ocean_proximity",axis = 1).columns.tolist() ##stores list of colums in csv without dropped one
    cat_attributes = ["ocean_proximity"]

    pipeline = build_pipeline(num_attributes,cat_attributes)
    housing_prepared = pipeline.fit_transform(housing_features)

    model = RandomForestRegressor(random_state=42) ##used as it is best in this dataset ,seen in previous file
    model.fit(housing_prepared,housing_lables)

    joblib.dump(model,MODEL_FILE) ##dumps the model into MODEL_FILE
    joblib.dump(pipeline,PIPELINE_FILE) ##dumps pipeline into PIPELINE_FILE
    print("model is trained congrats!")
else:
    #lets do interference
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)
    
    input_data = pd.read_csv("input.csv")
    transformed_input = pipeline.transform(input_data)
    predictions = model.predict(transformed_input)
    input_data['median_house_value'] = predictions

    input_data.to_csv("output.csv",index = False)
    print("Inference is Complete ,results saves to output.csv")