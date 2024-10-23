import warnings
import os
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, log_loss, r2_score
import xgboost as xgb
from Preprocessor_exe import Preprocessor
from XGB_optimizer_exe import XGB_optimizer
from SVC_optimizer_exe import SVC_Optimizer
from sklearn.naive_bayes import GaussianNB
import joblib

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

class Model_Executer:
    '''This class creates and optimizes three models: XGBoost (XGB), Support Vector Classifier (SVC), and Naive Bayes (NB).
    The models are optimized and trained on the provided dataset. The class also includes methods for saving these models.

    Attributes:
    -----------
    df : pd.DataFrame
        The input dataframe containing the data.
    df_imputed : pd.DataFrame
        The imputed dataframe after preprocessing.
    X_train : pd.DataFrame
        The training data features.
    X_test : pd.DataFrame
        The test data features.
    y_train : pd.Series
        The training data labels.
    y_test : pd.Series
        The test data labels.
    metrics_dict : dict
        A dictionary containing performance metrics for each model.

    Methods:
    --------
    preprocess(df, limit_try=3000, target="Label"):
        Preprocesses the input dataframe by dropping the "Timestamp" column and imputing missing values.
    split(df, label):
        Splits the dataframe into training and test sets.
    optimize_model(df, label="Label"):
        Optimizes and returns three models: XGBoost, SVC, and Naive Bayes.
    metrics(y_pred):
        Calculates and returns performance metrics (accuracy, RMSE, MAE, R2) for the predicted values.
    model_creator():
        Creates and trains the models, then returns a dictionary of performance metrics.
    saver(model_to_save, path, filename):
        Saves the specified model as a pickle file in the given path with a timestamped filename.
    '''
    
    def __init__(self, df, label: str = "Label") -> None:
        '''Initializes the Model_Executer class with the provided dataframe and target label.

        Parameters:
        ----------
        df : pd.DataFrame
            The input dataframe.
        label : str, optional
            The name of the target column to predict, default is "Label".
        '''
        self.df = df
        self.df_imputed = self.preprocess(self.df)
        self.X_train, self.X_test, self.y_train, self.y_test = self.split(self.df_imputed, label)
        self.metrics_dict = self.model_creator()

    def preprocess(self, df, limit_try: int = 3000, target: str = "Label"):
        '''Preprocesses the input dataframe by dropping the "Timestamp" column and imputing missing values.
        
        Parameters:
        ----------
        df : pd.DataFrame
            The input dataframe to preprocess.
        limit_try : int, optional
            The number of rows to use for initial preprocessing, default is 3000.
        target : str, optional
            The target column name, default is "Label".
        
        Returns:
        -------
        pd.DataFrame
            The imputed dataframe.
        '''
        if "Timestamp" in df.columns:
            df.drop("Timestamp", axis=1, inplace=True)
        
        df_veri = df.iloc[:limit_try, :]
        df_try = df.iloc[limit_try:, :]
            
        obj = Preprocessor(df=df_veri, target=target)
        df_imputed = obj.predictor(df_try)

        return df_imputed
            
    def split(self, df, label):
        '''Splits the dataframe into training and test sets.
        
        Parameters:
        ----------
        df : pd.DataFrame
            The input dataframe to split.
        label : str
            The name of the target column.
        
        Returns:
        -------
        tuple
            A tuple containing the training and test features and labels (X_train, X_test, y_train, y_test).
        '''
        X = df.drop(label, axis=1).copy()
        y = df[label]

        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

        return X_train, X_test, y_train, y_test

    def optimize_model(self, df: pd.DataFrame, label: str = "Label"):
        '''Optimizes and returns three models: XGBoost, SVC, and Naive Bayes.
        
        Parameters:
        ----------
        df : pd.DataFrame
            The imputed dataframe.
        label : str, optional
            The target column name, default is "Label".
        
        Returns:
        -------
        tuple
            A tuple containing the optimized models (XGBoost, SVC, Naive Bayes).
        '''

        def xgboost():
            print("Entering optimization 1")
            opti = XGB_optimizer(self.X_train, self.X_test, self.y_train, self.y_test)
            trial = opti.optimization()
            xgb_model = xgb.XGBClassifier(**trial.params)
            print(f"XGB model created with params: {xgb_model.get_params()}")
            return xgb_model
        
        def svc():
            print("Entering optimization 2")
            svc_o = SVC_Optimizer(self.X_train, self.X_test, self.y_train, self.y_test)
            best_svc_model = svc_o.svc_opt_model()
            print("SVC model created")
            return best_svc_model
        
        def naive_bayes():
            print("Entering optimization 3")
            gnb = GaussianNB()
            print("Naive Bayes model created") 
            return gnb
        
        xgb_o, svc_o, naive_bayes_o = xgboost(), svc(), naive_bayes()

        return xgb_o, svc_o, naive_bayes_o
    
    def metrics(self, y_pred):
        '''Calculates and returns performance metrics (accuracy, RMSE, MAE, R2) for the predicted values.
        
        Parameters:
        ----------
        y_pred : array-like
            The predicted values.
        
        Returns:
        -------
        dict
            A dictionary containing the calculated metrics.
        '''
        accuracy = round(accuracy_score(self.y_test, y_pred), 4)
        rmse = mean_squared_error(self.y_test, y_pred, squared=False)
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)

        metrics_dict = {"accuracy": accuracy, "rmse": rmse, "mae": mae, "r2": r2}

        return metrics_dict

    def model_creator(self):
        '''Creates and trains the models, then returns a dictionary of performance metrics.
        
        Returns:
        -------
        dict
            A dictionary containing performance metrics for each model.
        '''
        xgb_o, svc_o, naive_bayes_o = self.optimize_model(self.df_imputed, "Label")        

        metrics_dict = {}

        # Train and evaluate the XGBoost model
        xgb_model = xgb_o            
        booster = xgb_model.fit(self.X_train, self.y_train) 
        self.booster = booster
        y_pred_xgb = booster.predict(self.X_test)
        metrics_dict["XGB"] = self.metrics(y_pred_xgb)

        # Train and evaluate the SVC model
        svc = svc_o
        svc_fit = svc.fit(self.X_train, self.y_train)
        self.svc = svc_fit
        y_pred_svc = svc_fit.predict(self.X_test)
        metrics_dict["SVC"] = self.metrics(y_pred_svc)                

        # Train and evaluate the Naive Bayes model
        nb = naive_bayes_o
        nb_fit = nb.fit(self.X_train, self.y_train)
        self.nb = nb_fit
        y_pred_nb = nb_fit.predict(self.X_test)
        metrics_dict["NB"] = self.metrics(y_pred_nb)  

        return metrics_dict
    
    def saver(self, model_to_save, path, filename):
        '''Saves the specified model as a pickle file in the given path with a timestamped filename.
        
        Parameters:
        ----------
        model_to_save : object
            The model to be saved.
        path : str
            The directory path where the model will be saved.
        filename : str
            The name of the file to save the model as.
        '''
        cwd = os.getcwd()
        path = os.path.join(cwd, path)
        date = datetime.datetime.now().strftime("%d_%m_%Y")

        if os.path.exists(f"{path}\\{date}"):            
            try:                 
                joblib.dump(model_to_save, f"{path}\\{date}\\{filename}.pkl")                
            except:                
                pass
        else:          
            os.mkdir(f"{path}\\{date}") 
            print(f"Directory {path} created at {cwd}") 
            joblib.dump(model_to_save, f"{path}\\{date}\\{filename}.pkl")
