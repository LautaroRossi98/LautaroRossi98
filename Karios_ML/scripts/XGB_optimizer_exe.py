from functools import partial

# Import XGB classifier
import xgboost as xgb

# Import metrics to compute accuracy
from sklearn.metrics import accuracy_score

# Import optimizer
import optuna
import warnings

warnings.filterwarnings('ignore')

class XGB_optimizer:
    '''This class optimizes an XGBoost model using Optuna for hyperparameter tuning.
    It takes as input the result of a train-test split (X_train, X_test, y_train, y_test)
    and returns the best model possible based on accuracy.
    '''
    
    def __init__(self, X_train, X_test, y_train, y_test):
        '''Initializes the XGB_optimizer with training and test datasets.
        
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

    def objective(self, trial):
        '''Defines the objective function for Optuna to optimize XGBoost hyperparameters.
        
        Parameters:
        ----------
        trial : optuna.trial.Trial
            A single call of the objective function corresponds to one trial of the optimization.
        
        Returns:
        -------
        float
            The accuracy of the model with the current set of hyperparameters.
        '''
        # Define the hyperparameter search space
        param = {
            'objective': trial.suggest_categorical("obj", ['reg:squarederror', 'reg:logistic', 'multi:softmax']),
            'booster': trial.suggest_categorical("booster", ["gbtree", "dart"]),     
            'colsample_bynode': trial.suggest_float("colsample_bynode", 0.5, 1.0),
            'colsample_bytree': trial.suggest_float("colsample_bytree", 0.5, 1.0),        
            'n_estimators': 150,        
            'reg_alpha': trial.suggest_float("alpha", 1e-8, 1.0, log=True), # L1 regularization
            'reg_lambda': trial.suggest_float("lambda", 1e-8, 1.0, log=True), # L2 regularization        
            'subsample': trial.suggest_float("subsample", 0.2, 1.0),        
        }

        # Additional parameters specific to 'gbtree' or 'dart' boosters
        if param["booster"] in ["gbtree", "dart"]:
            param["max_depth"] = trial.suggest_int("max_depth", 3, 5, step=2)
            param["min_child_weight"] = trial.suggest_int("min_child_weight", 2, 10)
            param["eta"] =  trial.suggest_float("learning_rate", 0.008, 0.2)
            param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
            param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])

        if param["booster"] == "dart":
            param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
            param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
            param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
            param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)
        
        # Create and train the XGBoost model with the suggested hyperparameters
        xgb_model = xgb.XGBClassifier(**param)
        xgb_model.fit(self.X_train, self.y_train)
        
        # Predict and calculate accuracy
        y_pred = xgb_model.predict(self.X_test)
        accuracy = round(accuracy_score(self.y_test, y_pred), 4)

        return accuracy
    
    def optimization(self):
        '''Performs the optimization process using Optuna to find the best hyperparameters.
        
        Returns:
        -------
        optuna.trial.FrozenTrial
            The best trial found by the optimization process, containing the best hyperparameters and their corresponding accuracy.
        '''
        study = optuna.create_study(direction="maximize")
        study.optimize(partial(self.objective), n_trials=10, timeout=600)
        trial = study.best_trial
                
        return trial
