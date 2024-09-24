import logging

import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow.sklearn


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
parameters = parser.parse_args()

# adding logging
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    # import data and recode factor variables
    data = pd.read_csv("Housing.csv")

    data = pd.get_dummies(data, columns=['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning',
                                         'prefarea', 'furnishingstatus'])

    # prepare datasets for modelling
    training_data, testing_data = train_test_split(data, train_size=0.6, test_size=0.4)

    target_train = training_data[["price"]]
    target_test = testing_data[["price"]]

    vars_train = training_data.drop("price", axis=1)
    vars_test = testing_data.drop("price", axis=1)

    # parametrize model
    alpha = parameters.alpha
    l1_ratio = parameters.l1_ratio

    # adding Mlflow experiment
    exp = mlflow.set_experiment(experiment_name="first attempt")

    with mlflow.start_run(experiment_id=exp.experiment_id):
        # build model
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=111)
        model.fit(vars_train, target_train)

        # get predictions from model
        predicted_prices = model.predict(vars_test)

        # assess quality of model
        (rmse, mae, r2) = metrics(target_test, predicted_prices)

        print("Elasticnet model (alpha={:f}, l1_ratio={:f}):".format(alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        # log results and parameters
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        mlflow.sklearn.log_model(model, "elastic_net_with_mlflow")
