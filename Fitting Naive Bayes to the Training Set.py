    # Fitting Naive Bayes to the Training set  
    from sklearn.naive_bayes import GaussianNB  
    classifier = GaussianNB()  
    classifier.fit(x_train, y_train)  
