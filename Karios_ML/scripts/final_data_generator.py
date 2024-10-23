import warnings
import datasets
import datetime
import os
import pandas as pd

from sdv.evaluation.single_table import run_diagnostic, evaluate_quality
from sdv.evaluation.single_table import evaluate_quality, get_column_plot

# Metadata Creation API
from sdv.metadata import SingleTableMetadata
from sdv.lite import SingleTablePreset

warnings.filterwarnings("ignore")

class Synthetic_Data:
    '''This class is designed to create and evaluate synthetic data from a given dataframe. 
    It preprocesses the input data, generates metadata, trains a model, generates synthetic data, 
    and evaluates the quality of the synthetic data.
    
    Attributes:
    -----------
    df : pd.DataFrame
        The input dataframe containing the original data.
    df_metadata : SingleTableMetadata
        The metadata object generated from the dataframe.
    model : SingleTablePreset
        The trained model used to generate synthetic data.
    new_data : pd.DataFrame
        The generated synthetic data.
    
    Methods:
    --------
    registration(df):
        Preprocesses the input dataframe by removing the "Timestamp" column and resetting the index.
    save_new_data(df, path):
        Saves the provided dataframe to a CSV file at the specified path with a timestamped filename.
    metadata(validation_report=False):
        Generates metadata for the dataframe and optionally validates it.
    json_metadata():
        Saves the generated metadata in JSON format with a timestamped filename.
    model_train():
        Trains a model using the detected metadata.
    generator(n):
        Generates synthetic data using the trained model.
    evaluation(*args):
        Evaluates the quality of the synthetic data and generates column-wise plots for specified columns.
    '''

    def __init__(self, df):
        '''Initializes the Synthetic_Data class with the provided dataframe.
        
        Parameters:
        ----------
        df : pd.DataFrame
            The input dataframe to be used for generating synthetic data.
        '''
        self.df = self.registration(df)
        self.df_metadata = self.metadata()
        self.model = self.model_train()
        self.new_data = self.generator(n=500)

    def registration(self, df):
        '''Preprocesses the input dataframe by removing the "Timestamp" column and resetting the index.
        
        Parameters:
        ----------
        df : pd.DataFrame
            The input dataframe to be preprocessed.
        
        Returns:
        -------
        pd.DataFrame
            The preprocessed dataframe.
        '''
        if "Timestamp" in df.columns:
            df.drop("Timestamp", axis=1, inplace=True)
        
        df.reset_index(inplace=True)

        return df
    
    def save_new_data(self, df, path):
        '''Saves the provided dataframe to a CSV file at the specified path with a timestamped filename.
        
        Parameters:
        ----------
        df : pd.DataFrame
            The dataframe to be saved.
        path : str
            The directory path where the file will be saved.
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
            os.mkdir(path)
            print(f"Directory {path} created at {cwd}")
            df.to_csv(f"{path}\\{date}.csv")
    
    def metadata(self, validation_report: bool = False):
        '''Generates metadata for the dataframe and optionally validates it.
        
        Parameters:
        ----------
        validation_report : bool, optional
            If True, prints the validation report of the metadata, default is False.
        
        Returns:
        -------
        SingleTableMetadata
            The generated metadata object.
        '''
        metadata = SingleTableMetadata()        
        metadata.detect_from_dataframe(self.df)

        if validation_report:
            print(metadata.validate)
        
        return metadata
    
    def json_metadata(self):
        '''Saves the generated metadata in JSON format with a timestamped filename.'''
        self.df_metadata.save_to_json(filepath='synth_data_' + datetime.datetime.now().strftime("%d_%m_%Y-%H_%M_%S") + ".json")
    
    def model_train(self):
        '''Trains a model using a preset configuration and the detected metadata.
        
        Returns:
        -------
        SingleTablePreset
            The trained model used for generating synthetic data.
        '''
        synthesizer = SingleTablePreset(self.df_metadata, name='FAST_ML')
        synthesizer.fit(data=self.df)

        return synthesizer
    
    def generator(self, n: int):
        '''Generates synthetic data using the trained model.
        
        Parameters:
        ----------
        n : int
            The number of rows of synthetic data to generate.
        
        Returns:
        -------
        pd.DataFrame
            The generated synthetic data.
        '''
        synthetic_data = self.model.sample(num_rows=n)
        synthetic_data.drop("index", axis=1, inplace=True)
        
        return synthetic_data
    
    def evaluation(self, *args):
        '''Evaluates the quality of the synthetic data and generates column-wise plots for specified columns.
        
        Parameters:
        ----------
        *args : str
            Column names for which plots will be generated comparing real and synthetic data.
        '''
        quality_report = evaluate_quality(
                    self.df,
                    self.new_data,
                    self.df_metadata)
        
        for name in args:
            plot = get_column_plot(
                        real_data=self.df,
                        synthetic_data=self.new_data,
                        column_name=name,
                        metadata=self.df_metadata
                    )
                        
            plot.show('notebook')

# Posibles mejoras: unir el new_data a el original_df, generar una validación a través de los modelos de ML generados,
# hacer todo un EDA a través de los datos sintéticos, llamarlo de manera automática para hacer como si fuera una ingesta de datos automática (simil-producción)

# Equipo de API: conectar este generador a Evidently para ver si hay drifting, conectar el generador a los modelos de ML a ver cómo se comporta en producción
