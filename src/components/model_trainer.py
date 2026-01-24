import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor

from sklearn.ensemble import (AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor,)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evalute_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Splitting training and testing data")
            X_train,y_train,X_test,y_test =(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest" : RandomForestRegressor(),
                "Decisson Tree" : DecisionTreeRegressor(),
                "Gradient Boost" : GradientBoostingRegressor(),
                "Linear Regression" : LinearRegression(),
                "K-Neighbors Classifier" : KNeighborsRegressor(),
                "XGBoostClassifier" : XGBRegressor(),
                "Catboost Classifier" : CatBoostRegressor(),
                "Adaboost Classifier" : AdaBoostRegressor(),
            }

            model_report:dict=evalute_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)

            # to get best model 
            best_model_score = max(sorted(model_report.values()))

            # to get best model name from the dict 
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
                ]
            #pickup best model 

            best_model =   models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best Model found")
            logging.info(f"Best model found on both training and testing data")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted =best_model.predict(X_test)
            r2_squar = r2_score(y_test,predicted)

            return r2_squar


        except Exception as e:
            raise CustomException(e,sys)
            