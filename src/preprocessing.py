import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt;plt.rcdefaults()
import plotly.plotly as py
import plotly.graph_objs as graphobj
import plotly
plotly.tools.set_credentials_file(username='saxena-sagar', api_key='1HoemJdbXGB5YHKZ84GW')

class Preprocessing(object):
    """
    docstring forPreprocessing
    """

    def __init__(self, fname):
        self.fname = fname

    def read_normalise_dataset(self):
        '''
        The method should read a dataset and the argument to this method should be filename holding the datapoints to be read.
        '''
        read_data_set = pd.read_csv(self.fname)
        dataset_array = np.array(read_data_set)
        #selecting the two features from the X_train dataset
        dataset_selected_features = dataset_array[:,1:4]
        #print(dataset_array.shape)
        #appending y_train to x_train's selected features post analysis via PCA and creating a dataset to train the model
        y_train_read = pd.read_csv("/home/sagar/Desktop/seldon/datasets/y_train.csv")
        y_train_dataset_array = np.array(y_train_read)
        y_train_dataset = y_train_dataset_array[:,1:4]
        training_dataset = np.append(dataset_selected_features,y_train_dataset,axis=1)
        #print(training_dataset)

        #scaling the dataset array to remove outliers and keeing a low magnitude of the values
        scaled_dataset = preprocessing.scale(training_dataset)
        #Normalizing the dataset values
        dataset_normalized = preprocessing.normalize(scaled_dataset, norm='l2')
        #replacing the missing value from the dataset treated by numpy as NaN.
        # imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
        # imp.fit([[1, 2], [np.nan, 3], [7, 6]])
        #return dataset_normalized
        return training_dataset

    def pca_analysis_dataset(self,standardized_normalised_dataset):
        #calculating the mean vector where each value in this vector represents the sample mean of a feature column in the dataset
        dataset_mean_vec = np.mean(standardized_normalised_dataset, axis=0)
        #calculating the covariance matrix
        dataset_cov_matrix = (standardized_normalised_dataset - dataset_mean_vec).T.dot((standardized_normalised_dataset - dataset_mean_vec)) / (standardized_normalised_dataset.shape[0]-1)
        #print('Covariance matrix \n%s' %cov_mat)
        #calculating the eighen decomposition on the covariance matrix
        dataset_eig_vals, dataset_eig_vecs = np.linalg.eig(dataset_cov_matrix)
        #determining the eighen vectors with lowest eighen values to be dropped for a lower-dimensional subspace, by ranking eighenvalues from highest to lowest in order
        dataset_eig_pairs = [(np.abs(dataset_eig_vals[i]), dataset_eig_vecs[:,i]) for i in range(len(dataset_eig_vals))]
        #print("dataset_eig_pairs= \n",dataset_eig_pairs)
        #using explained variance to chose the major principle components
        total_variance_eig_vals = sum(dataset_eig_vals)
        expected_variance = [(i / total_variance_eig_vals)*100 for i in sorted(dataset_eig_vals, reverse = True)]
        cummulative_expected_variance = np.cumsum(expected_variance)

        #plotting bar and line graph to view principle components (of the features)
        principle_component_bars = graphobj.Bar(
        x=['PC %s' %i for i in range(1,20)],
        y=expected_variance,
        showlegend=False)

        cummulative_line_graph = graphobj.Scatter(
                x=['PC %s' %i for i in range(1,20)],
                y=cummulative_expected_variance,
                name='cumulative explained variance')

        graph_data = [principle_component_bars, cummulative_line_graph]

        graph_layout = graphobj.Layout(
                yaxis=dict(title='Explained variance in percent'),
                title='Explained variance by different principal components')
        fig = graphobj.Figure(data=graph_data, layout=graph_layout)
        py.iplot(fig)
        #Reducing the 20 dimension feature space to 2 dimension as per the PCA
        new_feature_space_matrix = np.hstack((dataset_eig_pairs[0][1].reshape(21,1),dataset_eig_pairs[1][1].reshape(21,1)))
        #print('new_feature_space_matrix:\n', new_feature_space_matrix)

        #projecting onto the new feature subspace
        new_feature_space = standardized_normalised_dataset.dot(new_feature_space_matrix)
        #print("new feature space size = ",new_feature_space.shape)

        # plt.plot(new_feature_space[0:5000,0], new_feature_space[0:5000,1], 'o', markersize=7, color='blue', alpha=0.5, label='class1')
        # plt.plot(new_feature_space[5000:10000,0], new_feature_space[5000:10000,1], '^', markersize=7, color='red', alpha=0.5, label='class2')
        #
        # plt.xlim([-0.9,0.9])
        # plt.ylim([-0.9,0.9])
        # plt.xlabel('x_values')
        # plt.ylabel('y_values')
        # plt.legend()
        # plt.title('New feature space post PCA with class labels')
        #
        # plt.show()
        return new_feature_space

def main():
    test = Preprocessing("/home/sagar/Desktop/seldon/datasets/X_train.csv")
    normalised_dataset = test.read_normalise_dataset()
    new_feature_dataset = test.pca_analysis_dataset(normalised_dataset)

if __name__ == '__main__':
    main()
