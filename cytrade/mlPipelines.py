import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor


def ttsplit(dset: pd.DataFrame, test_size):
    # Train-test split: 80% training 20% testing
    X_tr, X_tst, y_tr, y_tst = train_test_split(dset.drop(columns='target', axis=1), dset['target'], test_size=test_size, random_state=None, shuffle=False)
    print ("X_train:", X_tr.shape, "X_test:", X_tst.shape)
    print ("y_train:", y_tr.shape, "y_test:", y_tst.shape)
    return X_tr, X_tst, y_tr, y_tst


def feat_pipe1(dset: pd.DataFrame):
    # First Step
    cat_labels = ['Quarter', 'Month', 'Day_of_Week', 'Hour']  # 'Ticker', 'Market_State', 'Upcoming_Event'
    bool_labels = ['Is_Exp_Date','Is_Exp_DateTime']
    target_labels = ['target']

    cat_cols = (dset.columns.isin(cat_labels))
    bool_cols = (dset.columns.isin(bool_labels))
    num_cols = ~(dset.columns.isin(cat_labels + bool_labels + target_labels))

    cat_features = dset.loc[:,cat_cols]
    cat_names = [col for col in cat_features.columns]
    bool_features = dset.loc[:,bool_cols]
    bool_names = [col for col in bool_features.columns]

    num_features = dset.loc[:,num_cols]
    num_labels = [col for col in num_features.columns]

    trf1 = ColumnTransformer(transformers =[
        ('cat', SimpleImputer(strategy ='most_frequent'), cat_labels),
        ('bool', OneHotEncoder(drop='if_binary'), bool_labels),
        # ('bool', OrdinalEncoder(), bool_labels),
        ('num', SimpleImputer(strategy ='constant', fill_value=0), num_labels),
        
    ], remainder ='passthrough')

    # trf1.named_transformers_

    # Second Step
    # The column numbers to be transformed (here is the first n cols, as trf1 reordered them. Can also be a list, e.g. [0, 3, 5])
    cat_columns = list(range(len(cat_labels)))
    # You should drop dummy-encoded columns if using a linear model. Hence, sparse is set tp False.

    trf2 = ColumnTransformer(
        [('one_hot_encoder', OneHotEncoder(sparse=False, drop=None), cat_columns)],
        remainder='passthrough')  # The rest of the columns are excluded

    feat_pipe = Pipeline(steps =[
        ('tf1', trf1),
        ('tf2', trf2),
        ('tf3', RobustScaler(quantile_range=(25, 75))),  # or StandardScaler, MinMaxScaler etc.
        #('model', RandomForestRegressor(n_estimators = 20)),  # or LinearRegression, SVR, DecisionTreeRegressor, etc.
    ])

    return feat_pipe


def lda_pipe():
    mod = LinearDiscriminantAnalysis()
    model_pipe = Pipeline(steps =[
        ('model', mod),])
    return model_pipe
