import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# evaluation function
def metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


# passing arguments while running
parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, required=False, default=0.5)
parser.add_argument("--l1_ratio", type=float, required=False, default=0.5)
parameters  = parser.parse_args()

if __name__ == '__main__':
    data = pd.read_csv("Housing.csv")

    data = pd.get_dummies(data, columns=['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning',
                                         'prefarea', 'furnishingstatus'])
    training_data, testing_data = train_test_split(data, train_size=0.6, test_size=0.4)

    target_train = training_data[["price"]]
    target_test = testing_data[["price"]]

    vars_train = training_data.drop("price", axis=1)
    vars_test = testing_data.drop("price", axis=1)

    alpha = parameters.alpha
    l1_ratio = parameters.l1_ratio
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=111)
    model.fit(vars_train, target_train)

    predicted_prices = model.predict(vars_test)

    (rmse, mae, r2) = metrics(target_test, predicted_prices)

    print("Elasticnet model (alpha={:f}, l1_ratio={:f}):".format(alpha, l1_ratio))
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)
