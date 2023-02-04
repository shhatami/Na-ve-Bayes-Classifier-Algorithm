    # Visualising the Training set results  
    from matplotlib.colors import ListedColormap  
    x_set, y_set = x_train, y_train  
    X1, X2 = nm.meshgrid(nm.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step = 0.01),  
                         nm.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))  
    mtp.contourf(X1, X2, classifier.predict(nm.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),  
                 alpha = 0.75, cmap = ListedColormap(('purple', 'green')))  
    mtp.xlim(X1.min(), X1.max())  
    mtp.ylim(X2.min(), X2.max())  
    for i, j in enumerate(nm.unique(y_set)):  
        mtp.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],  
                    c = ListedColormap(('purple', 'green'))(i), label = j)  
    mtp.title('Naive Bayes (Training set)')  
    mtp.xlabel('Age')  
    mtp.ylabel('Estimated Salary')  
    mtp.legend()  
    mtp.show()  
