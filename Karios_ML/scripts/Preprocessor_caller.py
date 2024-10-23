import pandas as pd
from Preprocessor_exe import Preprocessor

class caller:
    '''
    This class serves as a wrapper to preprocess a dataframe and ensure that it is ready for further analysis.
    
    Attributes:
    ----------
    df : pd.DataFrame
        The processed dataframe after imputation.
    label : str
        The name of the target column to be predicted, default is "Label".
    '''
    
    def __init__(self, df: pd.DataFrame, label: str = "Label"):
        '''
        Initializes the caller object by processing the dataframe using the specified label column.
        
        Parameters:
        ----------
        df : pd.DataFrame
            The dataframe to be processed.
        label : str, optional
            The name of the target column, default is "Label".
        '''
        self.label = label
        self.df = self.processor(df, self.label) 
              

    def processor(self, df, label):
        '''
        Processes the dataframe by dropping the "Timestamp" column if it exists, 
        and then applies the Preprocessor to handle missing values.
        
        Parameters:
        ----------
        df : pd.DataFrame
            The dataframe to be processed.
        label : str
            The name of the target column to be used by the Preprocessor.
        
        Returns:
        -------
        pd.DataFrame
            The imputed dataframe with missing values handled.
        '''
        # Check if the dataframe contains a "Timestamp" column and remove it if present
        if "Timestamp" in df.columns:
            df.drop("Timestamp", axis=1, inplace=True)
       
        # Instantiate the Preprocessor with the dataframe and target column
        obj = Preprocessor(df = df, target = label)
        
        # Use the trained Preprocessor to fill missing values in the dataframe
        df_imputed = obj.predictor(df)

        return df_imputed


        




