from __future__ import annotations

import numpy as np
import pandas as pd
from citrees import CITreeRegressor
from lifelines import KaplanMeierFitter
from survival_data import SurvivalData


class ICTree:

    def __init__(self, survival_data_obj: SurvivalData, x: pd.DataFrame | np.matrix,
                 control=None, weights: list | pd.Series | None = None):

        """
        :param weights: a vector of weights for the survival data, usually set to None.
        :param survival_data_obj: a SurvivalData object
        :param x: pd.DataFrame containing the independent variables.
        :param control: dict object containing the control values entering the CItree regressor.
        """

        if control is None:
            control = {"min_samples_split": 2, "alpha": .05, "selector": 'pearson', "max_depth": -1, "max_feats": -1,
                       "n_permutations": 100, "early_stopping": False, "muting": True, "verbose": 0, "n_jobs": -1,
                       "random_state": None}
        self.control = control

        self.weights = weights
        self.data = survival_data_obj.survival_data.copy()
        self.control = control
        self.survival_data_obj = survival_data_obj
        self.x = x

    def ic_tree(self) -> CITreeRegressor:
        """
        The function fits a @citrees.CITreeRegressor with the interval-censored survival data, it uses
        ICTree.log_rank_transformation (which uses @lifelines.KaplanMeierFitter)in order to transform the survival data
        into a single vector log_rank as y.
        :return:A fitted citrees.CITreeRegressor
        """
        left = self.data.time1
        right = self.data.time2
        if sum((left - right) == 0) >= 1:
            unique_times = list(set(list(left.values) + list(right.values)))
            unique_times.sort()
            lag_unique_times = np.array(unique_times[1:])
            lead_unique_times = np.array(unique_times[:-1])
            epsilon = np.min(lag_unique_times - lead_unique_times) / 20
            self.data.loc[left == right, 'time2'] += epsilon
        if self.data.time1.isin([np.inf, -np.inf]):
            raise ValueError("Infinity values are in left side of the interval, make sure well defined and bounded")
        if np.isinf(self.data.time2).values.sum() == self.data.time2.shape[0]:
            raise ValueError("all values in right side of interval are Infinity, unable to compute")
        # replace infinity values with max value*100
        self.data.time2[np.isinf(right).values] = right[~np.isinf(right).values].max() * 100
        y = self.h2(self.weights)
        regressor = CITreeRegressor(**self.control)
        regressor.fit(X=self.x, y=y)
        return regressor

    def h2(self, weights: None | list | np.array | pd.Series):
        weights = np.array([1] * self.data.shape[0]) if weights is None else np.array(weights)
        log_rank = self.log_rank_transformation(self.data.loc[weights > 0].copy())
        r = pd.Series([0] * len(weights))
        r[weights > 0] = log_rank
        return np.matrix(r)

    @staticmethod
    def log_rank_transformation(survival_data: SurvivalData) -> pd.Series:
        """
        The function gets Interval censored survival data and returns log rank transformation using
         @lifelines.KaplanMeierFitter in order to get cumulative density and later on calculate the log rank.
        :return: pd.Series of log rank transformation
        """

        # Fit IC survival curve
        kmf = KaplanMeierFitter()
        kmf.fit_interval_censoring(survival_data['time1'], survival_data['time2'])
        # get estimated survival
        left_predicted = 1 - kmf.cumulative_density_at_times(survival_data['time1'])
        right_predicted = 1 - kmf.cumulative_density_at_times(survival_data['time2'])

        log_left = 0 if left_predicted <= 0 else left_predicted * np.log(left_predicted)
        log_right = 0 if right_predicted <= 0 else right_predicted * np.log(right_predicted)
        log_rank = (log_left - log_right) / (left_predicted - right_predicted)

        return log_rank
