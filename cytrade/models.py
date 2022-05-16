from river import linear_model, metrics, compose, preprocessing, feature_extraction, evaluate

from utils.flow import Kalman, get_hour, get_month, get_ordinal_date \
    , get_radial_basis_day, discretize_dt_day


class logisticReg(object):
    '''Stream learning Logistic Regression with river ML.'''
    def __init__(self,
                select=("o", "h", "l", "c", "v", "ret", "kal", "ama", "bbh", "bbm", "bbl"),
                extract=(get_ordinal_date, get_radial_basis_day),
                scale=preprocessing.StandardScaler,
                learn=linear_model.LogisticRegression):

        self.select = compose.Select(*select)
        self.extract_features = compose.TransformerUnion(*extract)
        self.scale = scale()
        self.learn =  learn()

        self.model = self.select 
        self.model += self.extract_features 
        self.model |= self.scale | self.learn