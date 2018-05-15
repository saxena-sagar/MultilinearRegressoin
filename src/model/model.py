from sklearn.model_selection import KFold
import sys
sys.path.append('./src')
from preprocessing import Preprocessing
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import mean_squared_error
import pandas as pd
import os

class Model(object):
    """docstModel."""
    def __init__(self, new_feature_file):
        self.new_feature_file = new_feature_file

    #making a split of 100 folds(datapoints) each in a split so that we have 10 slices
    def kfold_crossvalidation(self):
        number_of_folds = KFold(n_splits=10)
        #splitting the new dataset created post PCA into training set and test set depending on the k-fold value
        for train_index, test_index in number_of_folds.split(self.new_feature_file):
            new_feature_file_train, new_feature_file_test = self.new_feature_file[train_index], self.new_feature_file[test_index]
        return new_feature_file_train, new_feature_file_test

    #creating multivariate linear regression model
    def linearRegModel(self, training_dataset, test_dataset, dataset_selected_features):
        reg = LinearRegression()
        X = training_dataset[:,0:20]
        Y = training_dataset[:,20]
        reg.fit(X, Y)
        X_pred = test_dataset[:,0:20]
        Y_pred = test_dataset[:,20]
        y_pred = reg.predict(X_pred)
        print('RMSE Score: ', mean_squared_error(Y_pred, y_pred))
        #predicting y_test based on the x_test dataset provided
        X_test_array = np.array(pd.read_csv("./datasets/X_test.csv"))
        X_test = X_test_array[:,1:21]
        y_test = reg.predict(X_test)
        #writing y_test to .csv
        for i in range(y_test.size):
            d = {'index':range(10000),'target':y_test}
            df = pd.DataFrame(data = d,columns = ['index','target'])
            df.set_index('index', inplace=True)
        df.to_csv('./y_test.csv', encoding='utf-8')



def main():

    preprocessed_dataset = Preprocessing(os.path.abspath("./datasets/X_train.csv"))
    normalised_dataset,dataset_selected_features = preprocessed_dataset.read_normalise_dataset()
    model_creation = Model(normalised_dataset)
    train, test = model_creation.kfold_crossvalidation()
    model_training = model_creation.linearRegModel(train, test, dataset_selected_features)


if __name__ == '__main__':
    main()
