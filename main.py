from DataHandler import ImportData, DataInfo





def main():
    ### import the data ###
    ImportData.import_data()
    data_dict = ImportData.to_df()

    ### analyz the data ###
    analyzer = DataInfo(data_dict)
    #analyzer.salePrise_info()
    #analyzer.log_distributoin()
    #analyzer.get_missing_vals()
    #analyzer.features_coreleation()
    #analyzer.features_scatter("BedroomAbvGr", "HalfBath")
    #analyzer.categorical_features_analysis()
    


if __name__ == "__main__":
    main()