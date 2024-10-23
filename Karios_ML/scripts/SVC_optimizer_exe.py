# data processing libraries
from sklearn.model_selection import GridSearchCV
# import SVC classifier
from sklearn.svm import SVC

import warnings
warnings.filterwarnings('ignore')

class SVC_Optimizer:
    '''This class optimizes an SVC model by performing a GridSearch on fixed parameters.
    It takes as input the result of a train-test split (X_train, X_test, y_train, y_test)
    and returns the best model possible based on accuracy.
    '''
    
    def __init__(self, X_train, X_test, y_train, y_test):
        '''Initializes the SVC_Optimizer with training and test datasets.
        
        Parameters:
        ----------
        X_train : array-like
            Training feature data.
        X_test : array-like
            Test feature data.
        y_train : array-like
            Training target data.
        y_test : array-like
            Test target data.
        '''
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.parameters = [{'C': [1, 10, 100, 1000], 
                            'kernel': ['rbf'], 
                            'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]},                 
                           ]
        
    def optimizer(self):
        '''Performs GridSearch to find the best hyperparameters for the SVC model.
        
        Returns:
        -------
        dict
            The best parameters found during the GridSearch.
        '''
        svc = SVC()
        parameters = self.parameters  
        
        # Instantiate the GridSearchCV object with specified parameters
        grid_search = GridSearchCV(estimator=svc,  
                                   param_grid=parameters,
                                   scoring='accuracy',
                                   cv=5,
                                   refit=True,
                                   verbose=0,
                                   n_jobs=-1)

        # Fit the grid search to the training data
        grid_search.fit(self.X_train, self.y_train)
        
        # Return the best parameters found
        return grid_search.best_params_

    def svc_opt_model(self):
        '''Returns the optimized SVC model using the best parameters found by GridSearch.
        
        Returns:
        -------
        SVC
            An SVC model instance initialized with the best parameters.
        '''
        # Get the best parameters from the optimizer
        params = self.optimizer()
        
        # Initialize the SVC model with the best parameters
        svc = SVC(**params)
        
        # Print the best parameters for reference
        print(params)
        
        return svc

