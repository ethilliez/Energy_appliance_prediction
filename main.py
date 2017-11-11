from define_parameters import model_parameters, paths
import numpy as np
from numpy.random import seed
import logging
import pandas as pd 
import datetime
#
#from keras.layers import LSTM
#from keras import backend, regularizers, optimizers
#from keras.models import Sequential
#from tensorflow import set_random_seed
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error
#
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class model:
    def __init__(self):
        self.file = paths.FILE_PATH
        self.model = model_parameters.MODEL

    def read_data(self):
        logger.info(" Reading data...")
        DF = pd.read_csv(self.file)
        # Sort by date
        DF = DF.sort_values('date')
        # Remove random variables
        DF.pop('rv1')
        DF.pop('rv2')
        # Transform datetime variable into daily hours
        DF['date'] = pd.to_datetime(DF['date'])
        DF['hours'] = DF['date'].apply(lambda x: (x-datetime.datetime.combine(x.date(),datetime.time(0,0,0))).total_seconds())
        # Create features and label
        X = DF.drop('Appliances', axis =1).as_matrix()
        y = DF['Appliances'].as_matrix()
        print(DF.head(5))
        return X, y

    def standardize_data(self, X):
        logger.info(" Standardize the dataset...")
        X = (X-X.mean())/X.std()
        return X

    def split_train_test(self, X, y):
        logger.info(" Split training and testing dataset...")
        X_train = X[0:int(0.8*len(X)),:]
        X_test = X[int(0.8*len(X)):,:]
        y_train = y[0:int(0.8*len(y))]
        y_test = y[int(0.8*len(y)):]
        return X_train, y_train, X_test, y_test

    def train_model(self, X_train, y_train):
        logger.info(" Train model...")
        nb_samples = len(X_train)
        if(self.model == 'RandomForestRegressor'):
            regr = RandomForestRegressor()
            grid_values = {'max_depth': [4,5,6], 'min_samples_leaf': [int(0.005*nb_samples), int(0.01*nb_samples), int(0.03*nb_samples)], 'n_estimators': [50,75,100,125] }
        elif(self.model == 'SVR'):
            regr = SVR()
            grid_values = {'C': [10, 100, 250, 500, 1000], 'gamma': [0.1, 0.01, 0.001], 'kernel': ['rbf']}

        # Grid Search value when possible
        if(self.model in ['RandomForestRegressor', 'SVR']):
            grid_clf = GridSearchCV(regr, param_grid = grid_values, scoring ='r2', cv=3)
            grid_clf.fit(X_train, y_train)
            results = grid_clf.cv_results_['mean_test_score']
            logger.info(("  Best parameter: ", grid_clf.best_params_, "with best R2 score: ", round(grid_clf.best_score_,3)))
            #for result, params in zip(results, grid_clf.cv_results_['params']):
            #    print("%0.3f for %r" % (result, params))
            regr = grid_clf.best_estimator_
        return regr

    def test_model(self, X_test, regr, y_test):
        logger.info(" Test regression model...")
        y_predict = regr.predict(X_test)
        logger.info(("  R2 score: ", round(r2_score(y_test, y_predict),3)))
        logger.info(("  Mean Square error: ", round(mean_squared_error(y_test, y_predict),3)))
        return y_predict

    def plot(self, X_train, y_train, X_test, y_test, y_predict):
        logger.info(" Plot results...")
    	# Plot the results
        plt.figure()
        plt.scatter(X_train[:,0], y_train, s=20, edgecolor="black",
            c="darkorange", label="Train data")
        plt.scatter(X_test[:,0], y_test, c="darkgreen", edgecolor="black", label="Test data",  s=20)
        plt.plot(X_test[:,0], y_predict, color="yellowgreen", label="Predicted", linewidth=2)
        plt.xlabel("Datetime")
        plt.ylabel("Appliances usage")
        plt.title("Regression results")
        plt.legend()
        plt.savefig('Regression_plot.png', format='png', dpi =600)
        plt.show()

    def main(self):
        X, y = self.read_data()
        X = self.standardize_data(X[:,1:])  #[:,1:] to remove date infos
        X_train, y_train, X_test, y_test = self.split_train_test(X, y)
        regr = self.train_model(X_train[:,1:], y_train) 
        y_predict = self.test_model(X_test[:,1:],regr, y_test)
        self.plot(X_train, y_train, X_test, y_test, y_predict)

if __name__ == '__main__':
	process = model()
	process.main()
