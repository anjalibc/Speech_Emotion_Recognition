from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor

classification_grid_parameters = {
    SVC():  {
        'C': [0.0005, 0.001, 0.002, 0.01, 0.1, 1, 10],
        'gamma' : [0.001, 0.01, 0.1, 1],
        'kernel': ['rbf', 'poly', 'sigmoid']
    },
    GradientBoostingClassifier():   {
        'learning_rate': [0.05, 0.1, 0.3],
        'n_estimators': [40, 70, 100],
        'subsample': [0.3, 0.5, 0.7, 1],
        'min_samples_split': [0.2, 0.5, 0.7, 2],
        'min_samples_leaf': [0.2, 0.5, 1],
        'max_depth': [3, 7],
        'max_features': [1, 2, None],
    },
    MLPClassifier():    {
        'hidden_layer_sizes': [(200,), (300,), (400,), (128, 128), (256, 256)],
        'alpha': [0.001, 0.005, 0.01],
        'batch_size': [128, 256, 512, 1024],
        'learning_rate': ['constant', 'adaptive'],
        'max_iter': [200, 300, 400, 500]
    }
}

regression_grid_parameters = {
    # SVR():  {
    #     'C': [0.0005, 0.001, 0.002, 0.01, 0.1, 1, 10],
    #     'gamma' : [0.001, 0.01, 0.1, 1],
    #     'kernel': ['rbf', 'poly', 'sigmoid']
    # },
    GradientBoostingRegressor():   {
        'learning_rate': [0.05, 0.1, 0.3],
        'n_estimators': [40, 70, 100],
        'subsample': [0.3, 0.5, 0.7, 1],
        'min_samples_split': [0.2, 0.5, 0.7, 2],
        'min_samples_leaf': [0.2, 0.5, 1],
        'max_depth': [3, 7],
        'max_features': [1, 2, None],
    },
    MLPRegressor():    {
        'hidden_layer_sizes': [(200,), (200, 200), (300,), (400,)],
        'alpha': [0.001, 0.005, 0.01],
        'batch_size': [64, 128, 256, 512, 1024],
        'learning_rate': ['constant', 'adaptive'],
        'max_iter': [300, 400, 500, 600, 700]
    }
}