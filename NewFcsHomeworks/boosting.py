from __future__ import annotations

import math
from typing import List, Union, Tuple

from collections import defaultdict

from IPython.display import clear_output

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor


def score(clf, x, y):
    return roc_auc_score(y == 1, clf.predict_proba(x)[:, 1])


class MyDecisionTree:
    class Node:
        def __init__(self, answer: Union[float, None] = None, predicate: Union[float, None] = None, feature: Union[float, None] = None,
                     left: Union[MyDecisionTree.Node, None] = None,
                     right: Union[MyDecisionTree.Node, None] = None):
            """
            :param predicate: X_right[feature] >= predicate
            :param feature: X_right[feature] >= predicate
            :param answer: terminate node if None
            """
            self.answer = answer
            self.predicate = predicate
            self.feature = feature
            self.left = left
            self.right = right

    def __init__(self, loss: str = "mse", max_depth: int = None, feature_types: Union[List[str], None] = None,
                 min_samples_leaf: int = 1, min_samples_split: int = 2, features_count_rate: Union[float, None] = None):
        """
        :param loss: {"mse"} loss function
        :param feature_types: positional ist of {"cat", "num"} feature types
        """
        self.features_count_rate = features_count_rate
        self.root = None

        if max_depth is not None:
            assert max_depth >= 1, "max_depth must be >= 1"
        self.max_depth = max_depth

        assert min_samples_split >= 1, "min_samples_split must be >= 1"
        self.min_samples_leaf = min_samples_leaf

        assert min_samples_split >= 2, "min_samples_split must be >= 2"
        self.min_samples_split = min_samples_split

        if loss not in ["mse"]:
            raise ValueError(f"Wrong loss type {loss}")
        self.loss_type = loss

        if feature_types is not None:
            for i, feature_type in enumerate(feature_types):
                if feature_type not in ["cat", "num"]:
                    raise ValueError(f"wrong type of feature {i}")
        self.feature_types = feature_types

    def fit(self, X: np.ndarray, y: np.ndarray) -> MyDecisionTree:  # TODO input binarization
        assert X.shape[1] >= 1, "No empty datasets please"
        if self.feature_types is not None:
            assert X.shape[1] == len(self.feature_types), "Wrong dataset or feature_types size"
        assert y.shape[0] == X.shape[0], "Inputs must have the same size"
        depth = 1
        self.root = MyDecisionTree.Node()
        self.__fit_recursive(self.root, X, y, depth)
        return self

    def __fit_recursive(self, node: MyDecisionTree.Node, X: np.ndarray, y: np.ndarray, depth: int):
        if (self.max_depth is not None and depth == self.max_depth) or len(X) < self.min_samples_split:
            node.answer = self.__answer(y)
            return

        best_impurity = math.inf
        best_predicate = None
        best_feature = None

        features_count = int(X.shape[1] * self.features_count_rate) if self.features_count_rate is not None else X.shape[1]
        features_for_split = np.random.choice(np.arange(0, X.shape[1]), replace=False, size=(features_count, ))

        for feature in features_for_split:
            feature_type = "num"  # identify feature type
            if self.feature_types is not None:
                feature_type = self.feature_types[feature]

            match feature_type:
                case "num":
                    predicate, impurity = self.__get_numeric_best_split(X, y, feature)
                case "cat":
                    predicate, impurity = self.__get_categorical_best_split(X, y, feature)

            if impurity is not None and impurity < best_impurity:
                best_impurity = impurity
                best_predicate = predicate
                best_feature = feature

        if best_predicate is None:
            node.answer = self.__answer(y)
            return

        node.predicate = best_predicate
        node.feature = best_feature
        predicate_split = (X[:, best_feature] >= best_predicate)

        node.right = MyDecisionTree.Node()
        self.__fit_recursive(node.right, X[predicate_split], y[predicate_split], depth + 1)

        node.left = MyDecisionTree.Node()
        self.__fit_recursive(node.left, X[~predicate_split], y[~predicate_split], depth + 1)

    def __get_numeric_best_split(self, X: np.ndarray, y: np.ndarray, feature: int) -> Tuple[float, float]:
        feature_vector = X[:, feature]
        sorted_indices = np.argsort(feature_vector)
        return self.__find_best_split(feature_vector[sorted_indices], y[sorted_indices])

    def __get_categorical_best_split(self, X: np.ndarray, y: np.ndarray, feature: int) \
            -> Union[Tuple[float, float], Tuple[None, None]]:  # Optimal for gini, mse, logloss
        feature_vector = X[:, feature]

        categories, categories_count = np.unique(feature_vector, return_counts=True)
        categories_num = {cat: num for num, cat in enumerate(categories)}
        categories_rate = np.zeros(shape=(len(categories), ), dtype=np.double)

        for i in range(len(feature_vector)):
            cat = feature_vector[i]
            categories_rate[categories_num[cat]] += y[i]
        categories_rate /= categories_count

        sorted_indices = np.argsort(np.vectorize(lambda cat: categories_rate[categories_num[cat]])(feature_vector))
        return self.__find_best_split(feature_vector[sorted_indices], y[sorted_indices])

    def __find_best_split(self, feature_vector: np.ndarray, y: np.ndarray) \
            -> Union[Tuple[float, float], Tuple[None, None]]:
        """
        сплиты по последнему вхождению каждого элемента
        feature_vector: 1 1 2 3 4 4 4 5 5
        predicates: 1, 1.5, 2.5, 3.5, 4, 4, 4.5, 5
        last_occurrences: 1 2 3 6 8
        predicates[last_occurrences[:-1]] -> 1.5, 2.5, 3.5, 4.5
        """

        unique_values = np.unique(feature_vector)
        last_occurrences = np.searchsorted(feature_vector, unique_values, side="right") - 1
        lens = (last_occurrences + 1)[:-1]

        min_samples_leaf_filter = (lens >= self.min_samples_leaf) & (len(y) - lens >= self.min_samples_leaf)
        impurities = self.__impurity(y, last_occurrences[:-1])  # of dim = len(unique_values) - 1
        impurities = impurities[min_samples_leaf_filter]

        predicates = (feature_vector[1:] + feature_vector[:-1]) / 2
        predicates = predicates[last_occurrences[:-1]]
        predicates = predicates[min_samples_leaf_filter]

        if impurities.size:
            best_impurity_index = np.argmin(impurities)
            return predicates[best_impurity_index], impurities[best_impurity_index]
        else:
            return None, None

    def predict(self, X_test: np.ndarray) -> Union[np.ndarray, float]:   # Умеет выдавать ответ для одного объекта
        if X_test.ndim == 1:
            return self.__predict_x(X_test)
        else:
            preds = []
            for x in X_test:
                preds.append(self.__predict_x(x))
            return np.array(preds)

    def __predict_x(self, x: np.ndarray) -> float:
        node = self.root
        while node.answer is None:
            if x[node.feature] >= node.predicate:
                node = node.right
            else:
                node = node.left
        return node.answer

    def __impurity(self, y: np.ndarray, splits_index: np.ndarray) -> np.ndarray:
        lens = np.arange(1, len(y))

        left_losses = self.__get_all_splits_losses(y, lens)
        left_losses = left_losses[splits_index]

        right_losses = np.flip(self.__get_all_splits_losses(np.flip(y), lens))
        right_losses = right_losses[splits_index]

        lens = lens[splits_index]
        return (lens * left_losses + (len(y) - lens) * right_losses) / len(y)

    def __get_all_splits_losses(self, y: np.ndarray, lens: np.ndarray) -> np.ndarray:
        match self.loss_type:
            case "mse":
                cum_sum = np.cumsum(y)[:-1]
                quadratic_cum_sum = np.cumsum(y ** 2)[:-1]
                cum_means = cum_sum / lens
                losses = quadratic_cum_sum - 2 * cum_means * cum_sum + lens * (cum_means ** 2)
                return losses / lens
            case _:
                raise ValueError("wrong loss type")

    def __answer(self, y: np.ndarray) -> float:
        match self.loss_type:
            case "mse":
                return y.mean()
            case _:
                raise ValueError("wrong loss type")

    def eval(self, X: np.ndarray, y: np.ndarray) -> float:
        return self.__cals_loss(y, self.predict(X))

    def __cals_loss(self, y_true: np.ndarray, y_pred: Union[np.ndarray, None]) -> float:
        """
        :param y_pred: best_const if None
        :return: sum of losses on each sample
        """
        match self.loss_type:
            case "mse":
                return ((y_true - y_pred) ** 2).mean()
            case _:
                raise ValueError("wrong loss type")


class RandomForest:
    def __init__(self, n_estimators: int, base_model_params: Union[dict, None] = None,
                 features_count_rate: float = 0.33,  base_model_class=MyDecisionTree):
        self.n_estimators = n_estimators
        self.features_count_rate = features_count_rate

        self.base_model_class = base_model_class
        self.base_model_params: dict = {} if base_model_params is None else base_model_params

        self.models = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> RandomForest:
        for i in range(self.n_estimators):
            model = self.base_model_class(features_count_rate=self.features_count_rate)
            model.__dict__.update(self.base_model_params)
            bootstrap_index = np.random.choice(np.arange(0, len(X)), size=len(X))
            model.fit(X[bootstrap_index], y[bootstrap_index])  # TODO random features for every split
            self.models.append(model)
        return self

    def predict(self, X: np.ndarray) -> Union[np.ndarray, float]:  # Умеет выдавать ответ для одного объекта
        preds: Union[np.ndarray, float] = np.zeros(shape=(len(X), )) if X.ndim == 1 else 0
        for model in self.models:
            preds += model.predict(X)
        return preds / len(self.models)

    def eval(self, X: np.ndarray, y: np.ndarray):
        match self.base_model_params["loss"]:
            case "mse":
                return ((y - self.predict(X)) ** 2).mean()
            case _:
                raise ValueError("wrong loss type")


class GradientBoosting:
    def __init__(self, loss_type: str, n_estimators: int = 50, learning_rate: float = 0.1, base_model_params: Union[dict, None] = None,
                 subsample: float = 1.0, plot: bool = False, base_model_class = MyDecisionTree):

        if loss_type not in ["logloss", "mse"]:
            raise ValueError(f"wrong loss type")
        self.loss_type = loss_type

        assert n_estimators >= 1, f"Wtf, {n_estimators} n_estimators?"
        self.n_estimators = n_estimators

        assert 0.0 < subsample <= 1.0, "learning_rate must be between 0.0 and 1.0"
        self.learning_rate = learning_rate

        assert 0.0 < subsample <= 1.0, "subsample must be between 0.0 and 1.0"
        self.subsample = subsample  # TODO stochastic Gradient boosting (bootstrap)
        self.plot = plot

        self.base_model_class = base_model_class
        self.base_model_params = base_model_params if base_model_params is not None else {}
        self.models = []

        self.valid_loss_history = []
        self.train_loss_history = []

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_valid: Union[np.ndarray, None] = None, y_valid: Union[np.ndarray, None] = None) -> GradientBoosting:
        if self.loss_type == "lodloss":
            GradientBoosting.__check_classes(y_train)

        first_model = self.base_model_class(**self.base_model_params)  # tree of depth 1
        first_model.max_depth = 1
        self.models.append(first_model.fit(X_train, y_train))

        train_preds = self.models[0].predict(X_train)  # calc train_preds
        if X_valid is not None and y_valid is not None:
            val_preds = self.models[0].predict(X_valid)

        for i in range(self.n_estimators - 1):
            train_index = np.random.permutation(int(len(X_train) * self.subsample))  # случайная перестановка индексов
            model = self.__fit_base_model(X_train[train_index], y_train[train_index], train_preds[train_index])
            self.models.append(model)

            train_preds += self.learning_rate * model.predict(X_train)
            self.train_loss_history.append(self.__calc_metric(train_preds, y_train))

            if X_valid is not None and y_valid is not None:
                val_preds += self.learning_rate * model.predict(X_valid)
                self.valid_loss_history.append(self.__calc_metric(val_preds, y_valid))
            if self.plot:
                self.__plot(self.train_loss_history, self.valid_loss_history)

    def __fit_base_model(self,  X_train: np.ndarray, y: np.ndarray, train_preds: np.ndarray):
        model = self.base_model_class(**self.base_model_params)
        model.loss_type = "mse"
        base_model_y = self.__loss_anti_gradient(y, train_preds)
        return model.fit(X_train, base_model_y)

    def __loss_anti_gradient(self, y: np.ndarray, preds: np.ndarray) -> np.ndarray:  # свои производные на всех объектах
        match self.loss_type:
            case "logloss":
                # logloss = - ( yi * log(p(xi)) + (1 - yi) log(1 - p(xi)) )
                # - (p`(xi) * yi/ p(xi)  - p`(xi) * (1-yi) / (1 - p(xi)) )
                # - (yi * (1 - a(xi)) - (1 - yi) * a(xi))
                # gradient = - yi * (1 - a(xi)) + (1 - yi) * a(xi)
                preds = 1 / (1 + np.exp(-preds))
                return y * (1 - preds) - (1 - y) * preds
            case "mse":
                return - 2 * (preds - y)
            case _:
                raise ValueError(f"wrong loss type")

    def eval(self, X: np.ndarray, y: np.ndarray) -> float:
        return self.__calc_metric(self.predict(X), y)

    def __calc_metric(self, preds: np.ndarray, y: np.ndarray) -> float:
        match self.loss_type:
            case "logloss":
                # logloss = - 1/N * sum ( yi * log(a(xi)) + (1 - yi) log(1 - a(xi)) )
                preds = 1 / (1 + np.exp(-preds))
                return - np.mean(y * np.log(preds) + (1 - y) * np.log(1 - preds))
            case "mse":
                return np.mean((preds - y) ** 2)
            case _:
                raise ValueError(f"wrong loss type")

    def predict(self, X: np.ndarray) -> Union[np.ndarray, float]:  # Умеет выдавать ответ для одного объекта
        assert len(self.models) > 0, "model is not fitted"
        preds = self.models[0].predict(X)
        for i in range(1, len(self.models)):
            preds += self.learning_rate * self.models[i].predict(X)
        return preds

    def predict_proba(self, X: np.ndarray) -> Union[np.ndarray, float]:
        if self.loss_type == "logloss":
            pos_class_probs = 1 / (1 + np.exp(-self.predict(X)))
            return np.stack((pos_class_probs, 1 - pos_class_probs), axis=-1)
        else:
            raise ValueError(f"No predict proba for loss{self.loss_type}")

    def __plot(self, train_metrics: List[float], val_metrics: List[float]):
        clear_output()
        plt.plot(train_metrics, label="train")
        plt.plot(val_metrics, label="va")
        plt.xlabel("iteration")
        plt.ylabel(self.loss_type)
        plt.legend()
        plt.show()

    @staticmethod
    def __check_classes(y: np.ndarray):
        classes = np.unique(y)
        assert len(classes) == 2, "task is not binary classification"
        assert (classes[0] == 1 and classes[1] == 0) \
               or (classes[0] == 0 and classes[1] == 1), "classes must be 0 and 1"

# TODO удобные перепрогнозы
# TODO lf_leaf_reg (через перепрогнозы или без)
# TODO GB второго порядка
# TODO стоппить дерево, если отриц. impurity
# TODO catboost feature encoding

# class Boosting:
#
#     def __init__(
#             self,
#             base_model_params: dict = None,
#             n_estimators: int = 10,
#             learning_rate: float = 0.1,
#             subsample: float = 0.3,
#             early_stopping_rounds: int = None,
#             plot: bool = False,
#     ):
#         self.base_model_class = DecisionTreeRegressor
#         self.base_model_params: dict = {} if base_model_params is None else base_model_params
#
#         self.n_estimators: int = n_estimators
#
#         self.models: list = []
#         self.gammas: list = []
#
#         self.learning_rate: float = learning_rate
#         self.subsample: float = subsample
#
#         self.early_stopping_rounds: int = early_stopping_rounds
#         if early_stopping_rounds is not None:
#             self.validation_loss = np.full(self.early_stopping_rounds, np.inf)
#
#         self.plot: bool = plot
#
#         self.history = defaultdict(list)
#
#         self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
#         self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
#         self.loss_derivative = lambda y, z: -y * self.sigmoid(-y * z)
#
#     def fit_new_base_model(self, x, y, predictions):
#         self.gammas.append()
#         self.models.append()
#
#     def fit(self, x_train, y_train, x_valid, y_valid):
#         """
#         :param x_train: features array (train set)
#         :param y_train: targets array (train set)
#         :param x_valid: features array (validation set)
#         :param y_valid: targets array (validation set)
#         """
#         train_predictions = np.zeros(y_train.shape[0])
#         valid_predictions = np.zeros(y_valid.shape[0])
#
#         for _ in range(self.n_estimators):
#             self.fit_new_base_model()
#
#             if self.early_stopping_rounds is not None:
#                 pass
#
#         if self.plot:
#             pass
#
#     def predict_proba(self, x):
#         for gamma, model in zip(self.gammas, self.models):
#             pass
#
#     def find_optimal_gamma(self, y, old_predictions, new_predictions) -> float:
#         gammas = np.linspace(start=0, stop=1, num=100)
#         losses = [self.loss_fn(y, old_predictions + gamma * new_predictions) for gamma in gammas]
#
#         return gammas[np.argmin(losses)]
#
#     def score(self, x, y):
#         return score(self, x, y)
#
#     @property
#     def feature_importances_(self):
#         pass
