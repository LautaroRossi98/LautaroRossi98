import datasets
import pandas as pd
import matplotlib.pyplot as plt
import shap
import warnings
from model_creator_exe import Model_Executer
from Preprocessor_caller import caller
import numpy as np
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

class Explainers:
    '''This class provides various SHAP (SHapley Additive exPlanations) explanations for a given model.
    It allows for the generation of waterfall plots, mean SHAP value plots, beeswarm plots, and bar plots for model interpretation.
    '''

    def __init__(self, model, X_train: pd.DataFrame, category: list = list(range(0, 9))):
        '''Initializes the Explainers class with the model and training data.
        
        Parameters:
        ----------
        model : object
            The trained model to be explained.
        X_train : pd.DataFrame
            The training data used for model training.
        category : list, optional
            The list of categories or classes for which SHAP values are to be computed, default is list(range(0, 9)).
        '''
        self.model = model
        self.explainer = shap.Explainer(model)
        self.X_train = X_train
        self.category = category
        self.shap_values = self.explainer(X_train)
        self.means = self.shap_means_dataframe()
    
    def explainer_waterfall(self):
        '''Generates waterfall plots for each category specified in self.category using SHAP values.'''
        for i in self.category:
            shap.plots.waterfall(self.shap_values[0, :, i])
   
    def shap_means_dataframe(self):
        '''Computes the mean absolute SHAP values for each feature across all specified categories.
        
        Returns:
        -------
        pd.DataFrame
            A dataframe containing the mean SHAP values for each feature and category.
        '''
        pd_dict = {}

        for i in self.category:
            mean = f"mean_cat{i}"
            pd_dict[mean] = np.mean(np.abs(self.shap_values.values[:, :, i]), axis=0)
            
        df = pd.DataFrame(pd_dict)

        return df

    def means_plot(self):
        '''Plots the mean SHAP values as a bar plot for each feature.'''
        # Plot mean SHAP values
        fig, ax = plt.subplots(1, 1, figsize=(20, 10))
        self.means.plot.bar(ax=ax)

        ax.set_ylabel('Mean SHAP', size=30)
        ax.set_xticklabels(self.X_train.columns, rotation=90, size=10)
        ax.legend(fontsize=15)

    def bees(self, cat_class):
        '''Generates a beeswarm plot for a specific category/class using SHAP values.
        
        Parameters:
        ----------
        cat_class : int
            The category or class for which the beeswarm plot is to be generated.
        
        Raises:
        ------
        ValueError
            If the category is not an integer contained within the available labels specified in self.category.
        '''
        try:
            shap.plots.beeswarm(self.shap_values[:, :, cat_class])
        except:
            raise ValueError(f"Category must be an integer contained within the available labels {self.category}")
        
    def bars(self, cat_class):
        '''Generates a bar plot for a specific category/class using SHAP values.
        
        Parameters:
        ----------
        cat_class : int
            The category or class for which the bar plot is to be generated.
        
        Raises:
        ------
        ValueError
            If the category is not an integer contained within the available labels specified in self.category.
        '''
        try:
            shap.plots.bar(self.shap_values[:, :, cat_class], max_display=12)
        except:
            raise ValueError(f"Category must be an integer contained within the available labels {self.category}")
