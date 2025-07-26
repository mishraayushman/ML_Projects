import os 
import sys
from SOURCE.exception import CustomException
from SOURCE.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from SOURCE.components.data_transformation import DataTransformation
from SOURCE.components.data_transformation import DataTransformationConfig

@dataclass
class DataIngestionConfig:
    """
    This class is used to hold the paths for train ,test and raw data
    """
    train_data_path:str = os.path.join("artifacts",'train.csv')
    test_data_path:str = os.path.join('artifacts','test.csv')
    raw_data_path:str = os.path.join('artifacts','raw_data.csv')

class DataIngestion:
    def __init__(self,ingestion_config: DataIngestionConfig):
        self.ingestion_config = ingestion_config

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv("F:\\ML_Projects_new\\SOURCE\\Notebooks\\data\\stud.csv")
            logging.info("Reading Data....")
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info("test data split initated")

            train,test = train_test_split(df,test_size=0.2,random_state=42)

            train.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            
            logging.info('Data ingestion is completed')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            

        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ == '__main__':
    obj = DataIngestion(DataIngestionConfig())
    train_path, test_path = obj.initiate_data_ingestion()
    
    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_path,test_path)