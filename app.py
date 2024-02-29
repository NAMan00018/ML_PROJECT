from src.Mlproject.logger import logging
from src.Mlproject.exception import CustomException
from src.Mlproject.components.data_ingestion import DataIngestion
from src.Mlproject.components.data_ingestion import DataIngestionConfig
import sys

if __name__=="__main__":
    logging.info("execution has started")
    try:
        ##data_ingestion_config=DataIngestionConfig()
        data_ingestion=DataIngestion()
        data_ingestion.initiate_data_ingestion()
    except Exception as e:
        
        logging.info("Custom Exception")
        raise CustomException(e,sys)