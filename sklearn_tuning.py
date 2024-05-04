import keras_tuner

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split


SKLERAN_TRIALS_DIR = "."
SKLEARN_PROJECT_NAME = "sklearn_hp"


def build_hypermodel(hp):
    model_type = hp.Choice("model_type", values=["random_forest", "decision_tree", "ridge",])

    match model_type:
        case "random_forest":
            return RandomForestClassifier(
                n_estimators=hp.Int("n_estimators", 10, 50, step=10),
                max_depth=hp.Int("max_depth", 3, 10)
            )
        
        case "decision_tree":
            return DecisionTreeClassifier(
                criterion=hp.Choice("dense_activation", values=["gini", "entropy", "log_loss"]),
                max_depth=hp.Int("max_deptth", 4, 10, step=1)
            )

        case "ridge":
            return RidgeClassifier(
                alpha=hp.Float("alpha", 1e-3, 1, sampling="log")
            )
        case _:
            return None


if __name__ == "__main__":

    X, y = datasets.load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    optimizer = keras_tuner.oracles.BayesianOptimizationOracle(
            objective=keras_tuner.Objective("score", "max"),
            max_trials=100
        )

    tuner = keras_tuner.tuners.SklearnTuner(
        oracle=optimizer,
        hypermodel=build_hypermodel,
        scoring=make_scorer(accuracy_score),
        cv=StratifiedKFold(5),
        directory=SKLERAN_TRIALS_DIR,
        project_name=SKLEARN_PROJECT_NAME
    )

    tuner.search(X_train, y_train)

    best_model = tuner.get_best_models(num_models=1)[0]
    print(f"Best model: {best_model}")