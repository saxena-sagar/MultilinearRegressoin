from sklearn.model_selection import KFold
import sys
sys.path.insert(0,'/home/sagar/Desktop/seldon/src')
from preprocessing import Preprocessing
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import mean_squared_error
import pandas as pd

class Model(object):
    """docstModel."""
    def __init__(self, new_feature_file):
        self.new_feature_file = new_feature_file

    #making a split of 100 folds(datapoints) each in a split so that we have 10 slices
    def kfold_crossvalidation(self):
        number_of_folds = KFold(n_splits=10)
        #np.append(self.new_feature_file,)
        #number_of_folds.get_n_splits(self.new_feature_file)
        #splitting the new dataset created post PCA into training set and test set depending on the k-fold value
        for train_index, test_index in number_of_folds.split(self.new_feature_file):
            #print("TRAIN:", train_index, "TEST:", test_index)
            new_feature_file_train, new_feature_file_test = self.new_feature_file[train_index], self.new_feature_file[test_index]
            #print("TRAIN:", new_feature_file_train, "TEST:", new_feature_file_test)
        #print("new_feature_file_train=",new_feature_file_train)
        return new_feature_file_train, new_feature_file_test

    #creating multivariate linear regression model
    def linearRegModel(self, training_dataset, test_dataset):
        reg = LinearRegression()
        X = training_dataset[:,0:3]
        Y = training_dataset[:,2]
        reg.fit(X, Y)
        X_pred = test_dataset[:,0:3]
        Y_pred = test_dataset[:,2]
        y_pred = reg.predict(X_pred)
        #print('RMSE Score: ', mean_squared_error(Y_pred, y_pred))
        #predicting y_test based on the x_test dataset provided
        X_test_array = np.array(pd.read_csv("/home/sagar/Desktop/seldon/datasets/X_test.csv"))
        X_test = X_test_array[:,1:4]
        y_test = reg.predict(X_test)
        print("y_test: ",y_test)

def main():

    preprocessed_dataset = Preprocessing("/home/sagar/Desktop/seldon/datasets/X_train.csv")
    normalised_dataset = preprocessed_dataset.read_normalise_dataset()
    # #new_feature_dataset = preprocessed_dataset.pca_analysis_dataset(normalised_dataset)
    model_creation = Model(normalised_dataset)
    train, test = model_creation.kfold_crossvalidation()
    model_training = model_creation.linearRegModel(train, test)


if __name__ == '__main__':
    main()
