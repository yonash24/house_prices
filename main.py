from DataHandler import ImportData, DataInfo, DataCleaning, DataPreProcessing
from MlModel import Models




def main():
    ### import the data ###
    ImportData.import_data()
    data_dict = ImportData.to_df()

    ### analyz the data ###
    #analyzer = DataInfo(data_dict)
    #analyzer.salePrise_info()
    #analyzer.log_distributoin()
    #analyzer.get_missing_vals()
    #analyzer.features_coreleation()
    #analyzer.features_scatter("BedroomAbvGr", "HalfBath")
    #analyzer.categorical_features_analysis()
    
    ### clean the data ###
    cleaner = DataCleaning(data_dict)
    cleaner.fill_miss_vals()
    cleaner.outliers_handler()
    cleaner.remove_nois()

    ### preprocess the data ###
    clean_data = cleaner.data_dict
    proccesor = DataPreProcessing(clean_data)
    proccesor.features_engineer()
    proccesor.Skewness_handle()
    proccesor.encode_df()
    proccesor.data_standardize()

    x_train = proccesor.clean_data["X_train"]
    y_train = proccesor.clean_data["y_train"]
    test = clean_data["X_test"]

    ### create and train the models ###
    trainer = Models(x_train, test, y_train)
    trainer.choose_model()
    #its turnd out that ridge regression is the best one for us
    trainer.choose_model()
    trainer.ridge_model()
    trainer.XGB_model()
    


if __name__ == "__main__":
    main()