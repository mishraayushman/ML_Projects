import os
import sys

import numpy as np 
import pandas as pd
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from SOURCE.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(x_train,y_train,x_test,y_test,model):
    try:
        report = {}
        
        for i in range(len(list(model))):
            models = list(model.values())[i]
            models.fit(x_train,y_train)
            y_train_pred = models.predict(x_train)

            y_test_pred = models.predict(x_test)

            train_model_score = r2_score(y_train,y_train_pred)
            test_model_score = r2_score(y_test,y_test_pred)

            report[list(model.keys())[i]] = test_model_score
        
        return report
            


    except Exception as e:
        raise CustomException(e,sys)
