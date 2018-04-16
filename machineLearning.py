# This is a Python 3 script.

# Packages
def main():
    import numpy as np  # Linear algebra
    import pandas as pd  # Data pre-processing
    # %matplotlib inline
    import matplotlib.pyplot as plt
    from sklearn.model_selection import cross_val_score
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn import svm
    # from sklearn.svm import SVC
    # from sklearn.ensemble import RandomForestRegressor
    # from sklearn.linear_model import Ridge


    # Load data
    # Input data files are available in the same directory.

    train_df = pd.read_csv('train.csv', index_col=0)
    #test_df = pd.read_csv('Data/test.csv', index_col=0)
    #test_df = pd.read_csv('Data/test.csv')
    test_df = pd.read_csv('testFirebase.csv')

    # train_df = pd.read_csv('train.csv')
    # test_df = pd.read_csv('test.csv')
    test_df.insert(0, 'ID', range(4948, 4948 + len(test_df)))
    test_df = test_df.set_index('ID')
    #print(test_df.head(5))
    #test_df.to_csv('testData.csv', index=False)


    # train_df = pd.read_csv('train.csv')
    # test_df = pd.read_csv('test.csv')


    train_df.head(10)
    # test_df.head(10)



    ##1. Feature engineering & Data preprocessing


    # Combine test dataframe and train dataframe

    y_train = train_df.pop('label')
    all_df = pd.concat((train_df, test_df), axis=0)


    y_train.head(10)

    # Create dummy variables for all the catogorical features in the dataset

    all_dummy_df = pd.get_dummies(all_df)
    all_dummy_df.head(10)

    # all_df['MSSubClass'].dtypes
    # all_df['MSSubClass'] = all_df['MSSubClass'].astype(str)
    # all_df['MSSubClass'].value_counts()


    # Check the number of missing values for each feature

    all_dummy_df.isnull().sum().sort_values(ascending=False).head(20)

    # Get the mean values

    mean_cols = all_dummy_df.mean()
    mean_cols.head(20)

    # Replace missing values with mean values

    all_dummy_df = all_dummy_df.fillna(mean_cols)

    # Check the total number of missing values
    all_dummy_df.isnull().sum().sum()

    # Data normalization

    # Find out the numerical features in raw dataset

    numeric_cols = all_df.columns[all_df.dtypes != 'object']
    numeric_cols

    # z-score normalization

    numeric_col_means = all_dummy_df.loc[:, numeric_cols].mean()
    numeric_col_std = all_dummy_df.loc[:, numeric_cols].std()
    all_dummy_df.loc[:, numeric_cols] = (all_dummy_df.loc[:, numeric_cols] - numeric_col_means) / numeric_col_std

    # Check the values of variable after data normalization

    all_dummy_df['x'].hist();
    all_dummy_df['y'].hist();
    all_dummy_df['pressure'].hist();
    all_dummy_df['pitch'].hist();

    ## 2. Models


    # Split the dataframe into train and test

    dummy_train_df = all_dummy_df.loc[train_df.index]
    dummy_test_df = all_dummy_df.loc[test_df.index]
    dummy_train_df.shape, dummy_test_df.shape
    X_train = dummy_train_df.values
    X_test = dummy_test_df.values



    ## 3. Random Forest Prediction
    rf = RandomForestClassifier(n_estimators=200, max_features=.7)
    rf.fit(X_train, y_train)
    y_rf = rf.predict(X_test)
    y_final = y_rf
    print(y_final)
    #print(sum(y_final), len(y_final))
    #print(sum(y_final)/len(y_final))

    submission_df = pd.DataFrame(data={'Id': test_df.index, 'label': y_final})
    submission_df.head(10)
    submission_df.to_csv('submisison.csv', index=False)
    print('Testing done')

    if sum(y_final) / len(y_final) < 0.7:
        import sendEmail
        sendEmail.main()

if __name__ == "__main__":
    main()

