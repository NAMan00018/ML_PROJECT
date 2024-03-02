import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from dataclasses import dataclass
from sklearn.impute import SimpleImputer # in order to  handle categorical values/features and missing values
from sklearn.pipeline import Pipeline
from src.Mlproject.exception import CustomException
from src.Mlproject.logger import logging
from src.Mlproject.utils import save_object
import os # in order to save pickle file at a specific location

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class datatransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig() # so that we get the path in it
    def get_data_transformer_object(self):
        '''
        this function is responsible for data transformation
        '''
        try:
            numerical_columns=["writing_score","reading_score"]
            categorical_columns=[
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]
            numerical_pipeline=Pipeline(steps=[
                ("imputer",SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())
            ])
            categorical_pipeline=Pipeline(steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("onehotencoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
            ])
            logging.info(f"categorical_columns={categorical_columns}")
            logging.info(f"numerical_columns={numerical_columns}")
            preprocessor=ColumnTransformer(
                [
                    ("numerical_pipeline",numerical_pipeline,numerical_columns),
                    ("categorical_pipeline",categorical_pipeline,categorical_columns)
                ]
            )
            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("reading the train and test files")
            preprocessor_obj=self.get_data_transformer_object()
            target_column_name="math_score"
            numerical_column_name=["writing_score","reading_score"]

            ## dividing the train dataset into independent & dependent features

            input_features_train_df=train_df.drop(columns=[target_column_name],axis=1) # axis=1 means column wise removal
            target_feature_train_df=train_df[target_column_name]

             ## dividing the test dataset into independent & dependent features
            input_features_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info("applying preprocessing on training and testing dataset")

            input_Features_train_arr=preprocessor_obj.fit_transform(input_features_train_df)
            input_Features_test_arr=preprocessor_obj.transform(input_features_test_df)# dataleakage 
            train_arr=np.c_[
                input_Features_train_arr,np.array(target_feature_train_df)
            ]
            test_arr=np.c_[
                input_Features_test_arr,np.array(target_feature_test_df)
            ]
            logging.info("saved preprocessing object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )




        except Exception as e:
            raise CustomException(e,sys)
