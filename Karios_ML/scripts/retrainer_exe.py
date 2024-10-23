import xgboost as xgb
import warnings
from model_creator_exe import Model_Executer
from final_data_generator import Synthetic_Data
from Preprocessor_exe import Preprocessor
from sklearn.model_selection import train_test_split
from Preprocessor_caller import Caller
from XGB_optimizer_exe import XGB_optimizer
from SVC_optimizer_exe import SVC_Optimizer

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

class Retrainer:
    '''This class is designed to retrain a machine learning model using new data.
    It supports retraining for SVC, XGBoost, and Naive Bayes models.

    Attributes:
    -----------
    model : object
        The model to be retrained.
    label : str
        The name of the target column to be predicted.
    prep_data : pd.DataFrame
        The preprocessed dataframe after imputing missing values.
    x_train : pd.DataFrame
        The training data features.
    x_test : pd.DataFrame
        The test data features.
    y_train : pd.Series
        The training data labels.
    y_test : pd.Series
        The test data labels.
    retrained : object
        The retrained model.

    Methods:
    --------
    preprocessor(data, label):
        Preprocesses the input data using the Caller class.
    splitter():
        Splits the preprocessed data into training and test sets.
    retrain(model):
        Retrains and optimizes the given model using the training data.
    '''
    
    def __init__(self, data, model, label):
        '''Initializes the Retrainer class with the provided data, model, and target label.
        
        Parameters:
        ----------
        data : pd.DataFrame
            The input dataframe containing the new data.
        model : object
            The machine learning model to be retrained.
        label : str
            The name of the target column to predict.
        '''
        self.model = model
        self.label = label        
        self.prep_data = self.preprocessor(data, label)
        self.x_train, self.x_test, self.y_train, self.y_test = self.splitter()
        self.retrained = self.retrain(model)
        
    def preprocessor(self, data, label):
        '''Preprocesses the input data by imputing missing values.
        
        Parameters:
        ----------
        data : pd.DataFrame
            The input dataframe to preprocess.
        label : str
            The target column name.

        Returns:
        -------
        pd.DataFrame
            The preprocessed dataframe with imputed values.
        '''
        ca = Caller(data, label)
        return ca.df_imputed
    
    def splitter(self):
        '''Splits the preprocessed data into training and test sets.
        
        Returns:
        -------
        tuple
            A tuple containing the training and test features and labels (x_train, x_test, y_train, y_test).
        '''
        x_full = self.prep_data.drop(self.label, axis=1).copy()
        y_full = self.prep_data[self.label].copy()
        
        x_train, x_test, y_train, y_test = train_test_split(x_full, y_full, test_size=0.25)

        return x_train, x_test, y_train, y_test
    
    def retrain(self, model):
        '''Retrains and optimizes the provided model using the new training data.
        
        Parameters:
        ----------
        model : object
            The machine learning model to retrain.

        Returns:
        -------
        object
            The retrained model.
        '''
        if str(type(model)) == "<class 'sklearn.svm._classes.SVC'>":
            print("Retraining and optimizing SVC model")            
            svc_o = SVC_Optimizer(self.x_train, self.x_test, self.y_train, self.y_test)
            best_svc_model = svc_o.svc_opt_model()
            return best_svc_model.fit(self.x_train, self.y_train)
        
        elif str(type(model)) == "<class 'xgboost.sklearn.XGBClassifier'>":
            print("Retraining and optimizing XGB model")          
            opti = XGB_optimizer(self.x_train, self.x_test, self.y_train, self.y_test)
            trial = opti.optimization()
            xgb_model = xgb.XGBClassifier(**trial.params)
            return xgb_model.fit(self.x_train, self.y_train)
        
        elif str(type(model)) == "<class 'sklearn.naive_bayes.GaussianNB'>":
            print("Retraining and optimizing Naïve Bayes model")  
            return model.fit(self.x_train, self.y_train)

# Example usage:
# ret = Retrainer(model=me.booster, data=df, label="Label")
