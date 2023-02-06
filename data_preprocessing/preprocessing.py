import pandas as pd
import numpy as np
import datetime as dt
from imblearn.over_sampling import RandomOverSampler
class Preprocessor:
    """
        This class shall  be used to clean and transform the data before training.

        Written By: iNeuron Intelligence
        Version: 1.0
        Revisions: None

        """

    def __init__(self, file_object, logger_object):
        self.file_object = file_object
        self.logger_object = logger_object

    def remove_unwanted_spaces(self,data):
        """
                        Method Name: remove_unwanted_spaces
                        Description: This method removes the unwanted spaces from a pandas dataframe.
                        Output: A pandas DataFrame after removing the spaces.
                        On Failure: Raise Exception

                """
        self.logger_object.log(self.file_object, 'Entered the remove_unwanted_spaces method of the Preprocessor class')
        self.data = data

        try:
            self.df_without_spaces=self.data.apply(lambda x: x.str.strip() if x.dtype == "any" else x)  # drop the labels specified in the columns
            self.logger_object.log(self.file_object,
                                   'Unwanted spaces removal Successful.Exited the remove_unwanted_spaces method of the Preprocessor class')
            return self.df_without_spaces
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in remove_unwanted_spaces method of the Preprocessor class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'unwanted space removal Unsuccessful. Exited the remove_unwanted_spaces method of the Preprocessor class')
            raise Exception()   

    def separate_label_feature(self, data, label_column_name):
        """
                        Method Name: separate_label_feature
                        Description: This method separates the features and a Label Coulmns.
                        Output: Returns two separate Dataframes, one containing features and the other containing Labels .
                        On Failure: Raise Exception

                """
        self.logger_object.log(self.file_object, 'Entered the separate_label_feature method of the Preprocessor class')
        try:
            self.X=data.drop(labels=label_column_name,axis=1) # drop the columns specified and separate the feature columns
            self.Y=data[label_column_name] # Filter the Label columns
            self.logger_object.log(self.file_object,
                                   'Label Separation Successful. Exited the separate_label_feature method of the Preprocessor class')
            return self.X,self.Y
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in separate_label_feature method of the Preprocessor class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object, 'Label Separation Unsuccessful. Exited the separate_label_feature method of the Preprocessor class')
            raise Exception()

    def is_null_present(self,data):
        """
                                Method Name: is_null_present
                                Description: This method checks whether there are null values present in the pandas Dataframe or not.
                                Output: Returns True if null values are present in the DataFrame, False if they are not present and
                                        returns the list of columns for which null values are present.
                                On Failure: Raise Exception

                        """
        self.logger_object.log(self.file_object, 'Entered the is_null_present method of the Preprocessor class')
        self.null_present = False
        self.cols_with_missing_values=[]
        self.cols = data.columns
        try:
            self.null_counts=data.isna().sum() # check for the count of null values per column
            for i in range(len(self.null_counts)):
                if self.null_counts[i]>0:
                    self.null_present=True
                    self.cols_with_missing_values.append(self.cols[i])
            if(self.null_present): # write the logs to see which columns have null values
                self.dataframe_with_null = pd.DataFrame()
                self.dataframe_with_null['columns'] = data.columns
                self.dataframe_with_null['missing values count'] = np.asarray(data.isna().sum())
                self.dataframe_with_null.to_csv('preprocessing_data/null_values.csv') # storing the null column information to file
            self.logger_object.log(self.file_object,'Finding missing values is a success.Data written to the null values file. Exited the is_null_present method of the Preprocessor class')
            return self.null_present, self.cols_with_missing_values
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in is_null_present method of the Preprocessor class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object,'Finding missing values failed. Exited the is_null_present method of the Preprocessor class')
            raise Exception()

    def drop_missing_values(self, data, cols_with_missing_values):
        """
                                        Method Name: idrop_missing_values
                                        Description: This method drop all the missing values in the Dataframe using KNN Imputer.
                                        Output: A Dataframe which has all the missing values droped.
                                        On Failure: Raise Exception
                     """
        self.logger_object.log(self.file_object, 'Entered the missingvalues method of the Preprocessor class')
        self.data= data
        self.cols_with_missing_values=cols_with_missing_values
        try:
            
            self.data = self.data.dropna()
            self.logger_object.log(self.file_object, 'dropped missing values Successful.')
            return self.data
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in dropping missing_values method of the Preprocessor class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object,'Dropping missing values failed. Exited the missing_values method of the Preprocessor class')
            raise Exception()
    
   
    def encode_categorical_columns(self,data):
        """
                                                Method Name: encode_categorical_columns
                                                Description: This method encodes the categorical values to numeric values.
                                                Output: dataframe with categorical values converted to numerical values
                                                On Failure: Raise Exception

                             """
        self.logger_object.log(self.file_object, 'Entered the encode_categorical_columns method of the Preprocessor class')

        self.data=data
        try:
            self.cat_df = self.data.copy()
            self.cat_df['Age'] = 2015-self.cat_df['Year_Birth']
            self.cat_df['Dt_Customer'] = pd.to_datetime(self.cat_df['Dt_Customer'])
            self.cat_df['Month_Customer'] = 12.0 * (2015 - self.cat_df.Dt_Customer.dt.year ) + (1 - self.cat_df.Dt_Customer.dt.month)
            self.cat_df['TotalSpendings'] =  self.cat_df.MntWines + self.cat_df.MntFruits + self.cat_df.MntMeatProducts 
            + self.cat_df.MntFishProducts +self.cat_df.MntSweetProducts + self.cat_df.MntGoldProds
            self.cat_df.loc[(self.cat_df['Age'] >= 13) & (self.cat_df['Age'] <= 19), 'AgeGroup'] = 'Teen'
            self.cat_df.loc[(self.cat_df['Age'] >= 20) & (self.cat_df['Age']<= 39), 'AgeGroup'] = 'Adult'
            self.cat_df.loc[(self.cat_df['Age'] >= 40) & (self.cat_df['Age'] <= 59), 'AgeGroup'] = 'Middle Age Adult'
            self.cat_df.loc[(self.cat_df['Age'] > 60), 'AgeGroup'] = 'Senior Adult'
            self.cat_df['Children'] = self.cat_df['Kidhome'] + self.cat_df['Teenhome']
            self.cat_df.Marital_Status = self.cat_df.Marital_Status.replace({'Together': 'Partner',
                                                           'Married': 'Partner',
                                                           'Divorced': 'Single',
                                                           'Widow': 'Single', 
                                                           'Alone': 'Single',
                                                           'Absurd': 'Single',
                                                           'YOLO': 'Single'})
            self.cat_df = self.cat_df[self.cat_df.Age < 100]
            self.cat_df = self.cat_df[self.cat_df.Income < 120000]                                                                    

            self.data= self.cat_df.drop(['ID', 'Year_Birth', 'Education', 'Marital_Status', 'Kidhome', 'Teenhome', 'MntWines', 'MntFruits','MntMeatProducts',
                          'MntFishProducts', 'MntSweetProducts', 'MntGoldProds','Dt_Customer', 'Z_CostContact',
                          'Z_Revenue', 'Recency', 'NumDealsPurchases', 'NumWebPurchases','NumCatalogPurchases',
                          'NumStorePurchases', 'NumWebVisitsMonth', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5',
                          'AcceptedCmp1', 'AcceptedCmp2', 'Complain', 'AgeGroup'], axis=1)
            self.logger_object.log(self.file_object, 'encoding for categorical values successful. Exited the encode_categorical_columns method of the Preprocessor class')
            return self.data

        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in encode_categorical_columns method of the Preprocessor class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object, 'encoding for categorical columns Failed. Exited the encode_categorical_columns method of the Preprocessor class')
            raise Exception()

    def handle_imbalanced_dataset(self,x,y):
        """
        Method Name: handle_imbalanced_dataset
        Description: This method handles the imbalanced dataset to make it a balanced one.
        Output: new balanced feature and target columns
        On Failure: Raise Exception
                                     """
        self.logger_object.log(self.file_object,
                               'Entered the handle_imbalanced_dataset method of the Preprocessor class')

        try:
            self.rdsmple = RandomOverSampler()
            self.x_sampled,self.y_sampled  = self.rdsmple.fit_sample(x,y)
            self.logger_object.log(self.file_object,
                                   'dataset balancing successful. Exited the handle_imbalanced_dataset method of the Preprocessor class')
            return self.x_sampled,self.y_sampled

        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in handle_imbalanced_dataset method of the Preprocessor class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'dataset balancing Failed. Exited the handle_imbalanced_dataset method of the Preprocessor class')
            raise Exception()
