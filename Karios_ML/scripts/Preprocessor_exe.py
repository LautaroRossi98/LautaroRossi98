import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os

# Encoding libraries
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Visualization libraries
import missingno as msno

# Encoding missingness
import numpy as np

import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import layers 
import random as rd 

class Preprocessor:
    '''This class preprocesses the entire dataframe to ensure there are no missing values. 
    It uses a training dataset and returns an imputed dataframe. 
    The class assumes that categorical values are numerical booleans.
    When creating an instance of this class, pass the training dataset and the name of the target column.
    The resulting object will be a TensorFlow-trained imputer. To process a new dataframe, 
    use the Preprocessor.predictor(dataframe_to_fill) method.
    '''
    
    def __init__(self, df: pd.DataFrame, target: str = "Label"):
        '''Initializes the Preprocessor with the provided dataframe and target column.
        
        Parameters:
        ----------
        df : pd.DataFrame
            The dataframe to preprocess.
        target : str, optional
            The name of the target column, default is "Label".
        '''
        self.df = df
        self.target = target        
        self.complex_model = self.complexx(df, valid_percentage_100=20)
    
    def columns_cat(self, preprocessed: pd.DataFrame):
        '''Identifies categorical and numerical columns in the dataframe.
        
        Parameters:
        ----------
        preprocessed : pd.DataFrame
            The dataframe that has been preprocessed.
        
        Returns:
        -------
        tuple
            A tuple containing two lists: categorical and numerical columns.
        '''
        categorical = []
        numerical = []

        # Classify columns as categorical or numerical
        for col in preprocessed:
            if col == self.target:
                continue
            if preprocessed[col].nunique() <= 3:
                categorical.append(col)
            else:
                numerical.append(col)

        return categorical, numerical
    
    def make_noisy(self, np_data: pd.DataFrame):
        '''Adds noise to the dataset by shuffling the values in each column.
        
        Parameters:
        ----------
        np_data : pd.DataFrame
            The numpy array representing the dataframe to which noise will be added.
        
        Returns:
        -------
        np.array
            The noisy numpy array.
        '''
        np_ret = np.copy(np_data)
        
        # Shuffle the values in each column to create noise
        for i in range(np_ret.shape[1]):
            np.random.shuffle(np_ret[:,i])
        
        return np_ret  
    
    def preprocessor(self, numerical, x_fit):
        '''Prepares a preprocessing pipeline for numerical data using standard scaling.
        
        Parameters:
        ----------
        numerical : list
            List of numerical columns to be processed.
        x_fit : pd.DataFrame
            The dataframe to fit the preprocessing pipeline.
        
        Returns:
        -------
        ColumnTransformer
            A fitted ColumnTransformer that preprocesses numerical columns.
        '''
        # Preprocessing for numerical data
        numerical_transformer = Pipeline(verbose=False, steps=[
            ('scale', StandardScaler(with_mean=True, with_std=True)),
        ])        

        # Bundle preprocessing for numerical data
        preprocessor = ColumnTransformer(verbose=False,
            transformers=[
                ('pre_num', numerical_transformer, numerical),
            ])
        preprocessor.fit(x_fit)
                    
        return preprocessor
           
    def model_creator(self, input_dim):
        '''Creates and compiles a deep learning model for data imputation.
        
        Parameters:
        ----------
        input_dim : int
            The input dimension of the model (number of features).
        
        Returns:
        -------
        tuple
            A tuple containing the compiled Keras model and an early stopping callback.
        '''
        model_impute = keras.Sequential()
        model_impute.add(layers.Dense(20, activation='gelu', input_dim=input_dim, kernel_initializer='he_uniform'))
        model_impute.add(layers.Dense(16, activation='gelu', kernel_initializer='he_uniform'))
        model_impute.add(layers.Dense(10, activation='gelu', kernel_initializer='he_uniform', name='bottleneck'))
        model_impute.add(layers.Dense(16, activation='gelu', kernel_initializer='he_uniform'))
        model_impute.add(layers.Dense(20, activation='gelu', kernel_initializer='he_uniform'))
        model_impute.add(layers.Dense(input_dim, activation='linear', kernel_initializer='he_uniform'))

        optimizer = keras.optimizers.Adam(learning_rate=0.03)
        model_impute.compile(optimizer=optimizer, loss='msle')
        
        es = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', verbose=1, patience=50)

        return model_impute, es
    
    def get_numerical_summary(self):
        '''Generates a summary of missing values for numerical columns in the dataframe.
        
        Returns:
        -------
        dict or None
            A dictionary containing the percentage of missing values for each column,
            or None if there are no missing values.
        '''
        df = self.df
        total = df.shape[0]
        missing_columns = [col for col in df.columns if df[col].isnull().sum() > 0]
        missing_percent = {}
        
        # Calculate missing value percentages for each column
        for col in missing_columns:
            null_count = df[col].isnull().sum()
            per = (null_count / total) * 100
            missing_percent[col] = per
            print("{} : {} ({}%)".format(col, null_count, round(per, 3)))
        
        if not missing_percent:
            return None
            
        return missing_percent
    
    def statMissingValue(self):
        '''Provides statistics on missing values and zeros for each column in the dataframe.
        
        Returns:
        -------
        pd.DataFrame
            A dataframe containing statistics about missing values and zeros.
        '''
        lstSmry = []
        X = self.df

        # Gather statistics for each column
        for col in X.columns:
            zero = 0
            total = len(X.index)
            missing = X[col].isna().sum()
            missing_rate = round(missing * 100 / total, 2)
            nmer_unique = X[col].nunique()
            
            if X[col].dtype != 'object':
                zero = X[col].isin([0]).sum()
            
            zero_rate = round(zero * 100 / total, 2)
            lstSmry.append([col, str(X[col].dtype), total, nmer_unique, missing, missing_rate, zero, zero_rate])
        
        return pd.DataFrame(lstSmry, columns=["feature", "col_type", "total", "unique", "na", "na_rate", "zero", "zero_rate"])
    
    def simple(self, df_temp: pd.DataFrame, attribute_threshold: int = 25, sample_threshold: int = 5):
        '''Simple preprocessing that replaces missing values with medians for numerical columns.
        
        Parameters:
        ----------
        df_temp : pd.DataFrame
            The dataframe to be preprocessed.
        attribute_threshold : int, optional
            Not used in this function, default is 25.
        sample_threshold : int, optional
            Not used in this function, default is 5.
        
        Returns:
        -------
        pd.DataFrame
            The dataframe after simple preprocessing.
        '''
        df_temp = df_temp.copy()
        categorical = []
        numerical = []
        
        # columns
        missing_percent = self.get_numerical_summary()

        # Classify columns as categorical or numerical
        for col in df_temp.columns:
            if df_temp[col].nunique() <= 3:
                categorical.append(col)
            else:
                numerical.append(col)
       
        # Replace missing values in categorical columns with NaN
        for _ in categorical:
            df_temp.loc[:, _] = df_temp.loc[:, _].replace('', np.nan, regex=True)

        # Convert numerical columns to numeric and replace missing values with median
        for _ in numerical:     
            df_temp.loc[:, _] = df_temp.loc[:, _].replace('', np.nan, regex=True)
            df_temp.loc[:, _] = pd.to_numeric(df_temp.loc[:, _], errors='coerce')

        for _ in numerical:       
            df_temp.loc[:, _] = df_temp.loc[:, _].replace(np.nan, df_temp[_].median())
            df_temp.loc[:, _] = pd.to_numeric(df_temp.loc[:, _], errors='coerce')
        
        return df_temp
    
    def complexx(self, df: pd.DataFrame, valid_percentage_100: int = 20):
        '''Complex preprocessing that includes training a deep learning model for imputation.
        
        Parameters:
        ----------
        df : pd.DataFrame
            The dataframe to be preprocessed.
        valid_percentage_100 : int, optional
            The percentage of data to be used for validation, default is 20%.
        
        Returns:
        -------
        keras.Model
            The trained Keras model for imputation.
        '''
        df_preprocessed = self.simple(df)
        
        categorical, numerical = self.columns_cat(df_preprocessed)
                
        valid = round(df_preprocessed.shape[0] * valid_percentage_100 / 100)        

        # Shuffle the DataFrame 
        df = df_preprocessed.sample(frac=1).reset_index(drop=True)
        
        # Full dataset
        x_full = df[numerical].iloc[valid:, :].copy()
        
        columns = x_full.columns        
        
        # Validation dataset        
        x_valid = df[numerical].iloc[:valid, :].copy()
                
        y_full = df[self.target].iloc[valid:]

        # Splitting the data
        x_train, x_test, y_train, y_test = train_test_split(x_full, y_full, test_size=0.25)
        
        preprocessor = self.preprocessor(numerical, x_train)

        x_train_encoded = preprocessor.transform(x_train)
        x_test_encoded = preprocessor.transform(x_test)
        x_valid_encoded = preprocessor.transform(x_valid)
        
        input_dim = x_train_encoded.shape[1]

        model_impute, es = self.model_creator(input_dim)

        # Adding noise to the training data
        noise_X = self.make_noisy(x_train_encoded)  
        noise_X = np.concatenate((noise_X, np.copy(x_train_encoded)), axis=0)
        
        # Train the model on the noisy data
        his = model_impute.fit(noise_X, noise_X, epochs=2000, batch_size=512, shuffle=True, callbacks=[es], verbose=0)

        x_train_impute = pd.DataFrame(model_impute.predict(x_train_encoded), columns=columns)
        x_test_impute = pd.DataFrame(model_impute.predict(x_test_encoded), columns=columns)
        x_valid_impute = pd.DataFrame(model_impute.predict(x_valid_encoded), columns=columns)
        
        return model_impute
    
    def predictor(self, to_fill: pd.DataFrame):
        '''Predicts and imputes missing values in a new dataframe using the trained model.
        
        Parameters:
        ----------
        to_fill : pd.DataFrame
            The dataframe that needs missing value imputation.
        
        Returns:
        -------
        pd.DataFrame
            The dataframe with imputed numerical values merged with original categorical values.
        '''
        df_tof = to_fill
        
        df_preprocessed = self.simple(df_tof)
        
        categorical, numerical = self.columns_cat(df_preprocessed)
        preprocessor = self.preprocessor(numerical, df_tof)
        
        tofill = preprocessor.transform(df_tof)
        
        filled = pd.DataFrame(self.complex_model.predict(tofill), columns=numerical)

        # Merge the filled numerical features with the original categorical features
        merged_numerical = filled.merge(df_preprocessed[categorical], left_index=True, right_index=True)
        merged_numerical = merged_numerical.merge(df_tof["Label"], left_index=True, right_index=True)

        return merged_numerical
    
    def save_new_data(self, df: pd.DataFrame, path: str):
        '''Saves the processed dataframe to a CSV file in the specified directory.
        
        Parameters:
        ----------
        df : pd.DataFrame
            The dataframe to be saved.
        path : str
            The directory path where the dataframe should be saved.
        '''
        cwd = os.getcwd()
        path = os.path.join(cwd, path)
        date = datetime.datetime.now().strftime("%d_%m_%Y")

        if os.path.exists(path):
            try:                              
                df.to_csv(f"{path}\\{date}.csv")
            except:                
                pass
        else:
            path = os.path.join(cwd, path) 
            os.mkdir(f".{path}") 
            print(f"Directory {path} created at {cwd}")
            df.to_csv(f"{path}\\{date}.csv")
