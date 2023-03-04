from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

def evaluate(clf, X_train, y_train, X_val, y_val):
    # predict train
    y_train_hat = clf.predict(X_train)
    # compute R^2 metric
    acc_train = accuracy_score(y_train, y_train_hat)

    # predict val
    y_val_hat = clf.predict(X_val)
    # compute R^2 metric
    acc_val = accuracy_score(y_val, y_val_hat)

    # print performance
    print(f"Acc train: {acc_train}")
    print(f"Acc val : {acc_val}")

    print('Train performance')
    print('-------------------------------------------------------')
    print(classification_report(y_train, y_train_hat))

    print('Validation performance')
    print('-------------------------------------------------------')
    print(classification_report(y_val, y_val_hat))

    print('Roc_auc score')
    print('-------------------------------------------------------')
    print(roc_auc_score(y_val, y_val_hat))
    print('')

    print('Confusion matrix')
    print('-------------------------------------------------------')
    print(confusion_matrix(y_val, y_val_hat))

def save_test_preds(test, test_tf, preds):
    save_test = (test
        .join(test_tf.assign(Predicted = preds).Predicted)
        # missing values are entries where target == source node
        .assign(Predicted = lambda df_: df_.Predicted.mask(df_.Predicted.isna(), 1))
        # convert to int
        .assign(Predicted = lambda df_: df_.Predicted.astype(int))
        # remove useless columns
        .drop(["node1", "node2"], axis = 1, inplace = False)
    )

    # save predictions
    save_test.to_csv('data/test_preds.csv', index_label = "ID")

    return save_test