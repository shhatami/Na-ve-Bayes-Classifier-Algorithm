    # Importing the libraries  
    import numpy as nm  
    import matplotlib.pyplot as mtp  
    import pandas as pd  
      
    # Importing the dataset  
    dataset = pd.read_csv('user_data.csv')  
    x = dataset.iloc[:, [2, 3]].values  
    y = dataset.iloc[:, 4].values  
      
    # Splitting the dataset into the Training set and Test set  
    from sklearn.model_selection import train_test_split  
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)  
      
    # Feature Scaling  
    from sklearn.preprocessing import StandardScaler  
    sc = StandardScaler()  
    x_train = sc.fit_transform(x_train)  
    x_test = sc.transform(x_test)  
