import logging

import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet, Ridge, Lasso
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
parser.add_argument("--alpha", type=float, required=False, default=0.8)
parser.add_argument("--l1_ratio", type=float, required=False, default=0.8)
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

    # multiple experiments per file

################################ first experiment - elastic net ###################################################
    print("First Experiment Elastic Net")
    # adding Mlflow experiment
    exp = mlflow.set_experiment(experiment_name="first experiment")
    print("Name: {}".format(exp.name))
    print("Experiment id: {}".format(exp.experiment_id))


    #first run in first experiment
    mlflow.start_run(run_name="run1.1")

    tags = {
        "engineering": "ML platform",
        "release.candidate": "RC1",
        "release.version": "2.0"
    }

    mlflow.set_tags(tags)

    current_run = mlflow.active_run()
    print("Active run id is {}".format(current_run.info.run_id))
    print("Active run name is {}".format(current_run.info.run_name))

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
    metrics_dict = {
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }
    mlflow.log_metrics(metrics_dict)
    mlflow.sklearn.log_model(model, "elastic_net_with_mlflow")
    mlflow.log_artifacts("data/")
    artifacts_uri = mlflow.get_artifact_uri()
    print("The artifact path is", artifacts_uri)
    mlflow.end_run()

    #second run in first experiment
    mlflow.start_run(run_name="run2.1")

    tags = {
        "engineering": "ML platform",
        "release.candidate": "RC1",
        "release.version": "2.0"
    }

    mlflow.set_tags(tags)

    current_run = mlflow.active_run()
    print("Active run id is {}".format(current_run.info.run_id))
    print("Active run name is {}".format(current_run.info.run_name))

    # build model
    model = ElasticNet(alpha=0.9, l1_ratio=0.9, random_state=111)
    model.fit(vars_train, target_train)

    # get predictions from model
    predicted_prices = model.predict(vars_test)

    # assess quality of model
    (rmse, mae, r2) = metrics(target_test, predicted_prices)

    print("Elasticnet model (alpha={:f}, l1_ratio={:f}):".format(0.9, 0.9))
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)

    # log results and parameters
    mlflow.log_param("alpha", 0.9)
    mlflow.log_param("l1_ratio", 0.9)
    metrics_dict = {
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }
    mlflow.log_metrics(metrics_dict)
    mlflow.sklearn.log_model(model, "elastic_net_with_mlflow")
    mlflow.log_artifacts("data/")
    artifacts_uri = mlflow.get_artifact_uri()
    print("The artifact path is", artifacts_uri)
    mlflow.end_run()

    #third run in first experiment
    mlflow.start_run(run_name="run3.1")

    tags = {
        "engineering": "ML platform",
        "release.candidate": "RC1",
        "release.version": "2.0"
    }

    mlflow.set_tags(tags)

    current_run = mlflow.active_run()
    print("Active run id is {}".format(current_run.info.run_id))
    print("Active run name is {}".format(current_run.info.run_name))

    # build model
    model = ElasticNet(alpha=0.4, l1_ratio=0.4, random_state=111)
    model.fit(vars_train, target_train)

    # get predictions from model
    predicted_prices = model.predict(vars_test)

    # assess quality of model
    (rmse, mae, r2) = metrics(target_test, predicted_prices)

    print("Elasticnet model (alpha={:f}, l1_ratio={:f}):".format(0.4, 0.4))
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)

    # log results and parameters
    mlflow.log_param("alpha", 0.4)
    mlflow.log_param("l1_ratio", 0.4)
    metrics_dict = {
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }
    mlflow.log_metrics(metrics_dict)
    mlflow.sklearn.log_model(model, "elastic_net_with_mlflow")
    mlflow.log_artifacts("data/")
    artifacts_uri = mlflow.get_artifact_uri()
    print("The artifact path is", artifacts_uri)
    mlflow.end_run()

######################## second experiment - ridge ##################################################3
    print("Second Experiment Ridge")
    # adding Mlflow experiment
    exp = mlflow.set_experiment(experiment_name="first exp_multi_Ridge")
    print("Name: {}".format(exp.name))
    print("Experiment id: {}".format(exp.experiment_id))

    # first run in second experiment
    mlflow.start_run(run_name="run1.1")

    tags = {
        "engineering": "ML platform",
        "release.candidate": "RC1",
        "release.version": "2.0"
    }

    mlflow.set_tags(tags)

    current_run = mlflow.active_run()
    print("Active run id is {}".format(current_run.info.run_id))
    print("Active run name is {}".format(current_run.info.run_name))

    # build model
    model = Ridge(alpha=alpha, random_state=111)
    model.fit(vars_train, target_train)

    # get predictions from model
    predicted_prices = model.predict(vars_test)

    # assess quality of model
    (rmse, mae, r2) = metrics(target_test, predicted_prices)

    print("Ridge model (alpha={:f}:".format(alpha))
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)

    # log results and parameters
    mlflow.log_param("alpha", alpha)
    metrics_dict = {
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }
    mlflow.log_metrics(metrics_dict)
    mlflow.sklearn.log_model(model, "ridge_with_mlflow")
    mlflow.log_artifacts("data/")
    artifacts_uri = mlflow.get_artifact_uri()
    print("The artifact path is", artifacts_uri)
    mlflow.end_run()

    # second run in second experiment
    mlflow.start_run(run_name="run2.1")

    tags = {
        "engineering": "ML platform",
        "release.candidate": "RC1",
        "release.version": "2.0"
    }

    mlflow.set_tags(tags)

    current_run = mlflow.active_run()
    print("Active run id is {}".format(current_run.info.run_id))
    print("Active run name is {}".format(current_run.info.run_name))

    # build model
    model = Ridge(alpha=0.9, random_state=111)
    model.fit(vars_train, target_train)

    # get predictions from model
    predicted_prices = model.predict(vars_test)

    # assess quality of model
    (rmse, mae, r2) = metrics(target_test, predicted_prices)

    print("Ridge model (alpha={:f}:".format(0.9))
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)

    # log results and parameters
    mlflow.log_param("alpha", 0.9)
    metrics_dict = {
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }
    mlflow.log_metrics(metrics_dict)
    mlflow.sklearn.log_model(model, "ridge_with_mlflow")
    mlflow.log_artifacts("data/")
    artifacts_uri = mlflow.get_artifact_uri()
    print("The artifact path is", artifacts_uri)
    mlflow.end_run()

    # third run in second experiment
    mlflow.start_run(run_name="run3.1")

    tags = {
        "engineering": "ML platform",
        "release.candidate": "RC1",
        "release.version": "2.0"
    }

    mlflow.set_tags(tags)

    current_run = mlflow.active_run()
    print("Active run id is {}".format(current_run.info.run_id))
    print("Active run name is {}".format(current_run.info.run_name))

    # build model
    model = Ridge(alpha=0.4, random_state=111)
    model.fit(vars_train, target_train)

    # get predictions from model
    predicted_prices = model.predict(vars_test)

    # assess quality of model
    (rmse, mae, r2) = metrics(target_test, predicted_prices)

    print("Ridge model (alpha={:f}:".format(0.4))
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)

    # log results and parameters
    mlflow.log_param("alpha", 0.4)
    metrics_dict = {
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }
    mlflow.log_metrics(metrics_dict)
    mlflow.sklearn.log_model(model, "ridge_with_mlflow")
    mlflow.log_artifacts("data/")
    artifacts_uri = mlflow.get_artifact_uri()
    print("The artifact path is", artifacts_uri)
    mlflow.end_run()


######################## third experiment - lasso ##################################################3
    print("Third Experiment Lasso")
    # adding Mlflow experiment
    exp = mlflow.set_experiment(experiment_name="first exp_multi_lasso")
    print("Name: {}".format(exp.name))
    print("Experiment id: {}".format(exp.experiment_id))

    # first run in third experiment
    mlflow.start_run(run_name="run1.1")

    tags = {
        "engineering": "ML platform",
        "release.candidate": "RC1",
        "release.version": "2.0"
    }

    mlflow.set_tags(tags)

    current_run = mlflow.active_run()
    print("Active run id is {}".format(current_run.info.run_id))
    print("Active run name is {}".format(current_run.info.run_name))

    # build model
    model = Lasso(alpha=alpha, random_state=111)
    model.fit(vars_train, target_train)

    # get predictions from model
    predicted_prices = model.predict(vars_test)

    # assess quality of model
    (rmse, mae, r2) = metrics(target_test, predicted_prices)

    print("Lasso model (alpha={:f}:".format(alpha))
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)

    # log results and parameters
    mlflow.log_param("alpha", alpha)
    metrics_dict = {
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }
    mlflow.log_metrics(metrics_dict)
    mlflow.sklearn.log_model(model, "lasso_with_mlflow")
    mlflow.log_artifacts("data/")
    artifacts_uri = mlflow.get_artifact_uri()
    print("The artifact path is", artifacts_uri)
    mlflow.end_run()

    # second run in third experiment
    mlflow.start_run(run_name="run2.1")

    tags = {
        "engineering": "ML platform",
        "release.candidate": "RC1",
        "release.version": "2.0"
    }

    mlflow.set_tags(tags)

    current_run = mlflow.active_run()
    print("Active run id is {}".format(current_run.info.run_id))
    print("Active run name is {}".format(current_run.info.run_name))

    # build model
    model = Lasso(alpha=0.9, random_state=111)
    model.fit(vars_train, target_train)

    # get predictions from model
    predicted_prices = model.predict(vars_test)

    # assess quality of model
    (rmse, mae, r2) = metrics(target_test, predicted_prices)

    print("Lasso model (alpha={:f}:".format(0.9))
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)

    # log results and parameters
    mlflow.log_param("alpha", 0.9)
    metrics_dict = {
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }
    mlflow.log_metrics(metrics_dict)
    mlflow.sklearn.log_model(model, "lasso_with_mlflow")
    mlflow.log_artifacts("data/")
    artifacts_uri = mlflow.get_artifact_uri()
    print("The artifact path is", artifacts_uri)
    mlflow.end_run()

    # third run in third experiment
    mlflow.start_run(run_name="run3.1")

    tags = {
        "engineering": "ML platform",
        "release.candidate": "RC1",
        "release.version": "2.0"
    }

    mlflow.set_tags(tags)

    current_run = mlflow.active_run()
    print("Active run id is {}".format(current_run.info.run_id))
    print("Active run name is {}".format(current_run.info.run_name))

    # build model
    model = Lasso(alpha=0.4, random_state=111)
    model.fit(vars_train, target_train)

    # get predictions from model
    predicted_prices = model.predict(vars_test)

    # assess quality of model
    (rmse, mae, r2) = metrics(target_test, predicted_prices)

    print("Lasso model (alpha={:f}:".format(0.4))
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)

    # log results and parameters
    mlflow.log_param("alpha", 0.4)
    metrics_dict = {
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }
    mlflow.log_metrics(metrics_dict)
    mlflow.sklearn.log_model(model, "lasso_with_mlflow")
    mlflow.log_artifacts("data/")
    artifacts_uri = mlflow.get_artifact_uri()
    print("The artifact path is", artifacts_uri)
    mlflow.end_run()

    run = mlflow.last_active_run()
    print("Recent Active run id is {}".format(run.info.run_id))
    print("Recent Active run name is {}".format(run.info.run_name))