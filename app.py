from src.Mlproject.logger import logging
from src.Mlproject.exception import CustomException
from src.Mlproject.components.data_ingestion import DataIngestion
from src.Mlproject.components.data_ingestion import DataIngestionConfig
from src.Mlproject.components.data_transformation import datatransformation
import sys

if __name__=="__main__":
    logging.info("execution has started")
    try:
        ##data_ingestion_config=DataIngestionConfig()
        data_ingestion=DataIngestion()
        train_data_path,test_data_path=data_ingestion.initiate_data_ingestion()

        data_transformation=datatransformation()
        data_transformation.initiate_data_transformation(train_data_path,test_data_path)
    except Exception as e:
        
        logging.info("Custom Exception")
        raise CustomException(e,sys)