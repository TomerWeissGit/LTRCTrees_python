from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor

from survival_data import SurvivalData


class LTRCart:
    def __init__(self, survival_data_obj: SurvivalData, x: pd.DataFrame, weights: Optional[tuple | list] = None,
                 number_of_se_from_ccp: Optional[float] = 0.0, control: Optional[dict] = None):
        """

        :param survival_data_obj: SurvivalData type object (contains time1,time2,status)
        :param x: explaining variables pd.DataFrame to fit the decision tree regressor
        :param weights: tuple or list of weights to fit the decision tree regressor,
        see sklearn.tree.DecisionTreeRegressor.fit() param for more info.
        :param number_of_se_from_ccp: number of cross validation SE willing to take from the best ccp got in cv.
        :param control: control params to fit into the DecisionTreeRegressor, see sklearn.tree.DecisionTreeRegressor
        for more info. should be a dictionary containing the param name as key and param value as value.
        """
        self.data = survival_data_obj.survival_data.copy()
        self.weights = weights
        self.number_of_se_from_ccp = number_of_se_from_ccp
        self.control = control
        if control is None:
            self.control = dict()
        self.x = x

    def create_survival_data(self) -> pd.DataFrame:
        """
        :return: pd.DataFrame with an event and time columns where the event column is equal to the status column of the
        survival_data_obj and the time column is the diff between the transformation from coxPH model prediction and
        then linear interpolation of those predictions for time1 and time2 of the survival_data_obj entered.,
        """
        y = self.data
        status = self.data.event.copy()
        times = self.data.time2.copy()
        unique_death_times = np.sort(times[status == 1].unique())
        cox_ph = CoxPHFitter(penalizer=0)
        cox_baseline_cum_hazard = (cox_ph.fit(df=y, duration_col='time2', event_col='event', entry_col='time1')
                                   .baseline_cumulative_hazard_)
        # remove inf hazard and Nones.
        cox_baseline_cum_hazard = cox_baseline_cum_hazard.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
        cox_baseline_hazard_death = cox_baseline_cum_hazard.loc[cox_baseline_cum_hazard.index.isin(unique_death_times)]
        cum_haz_times = [0] + list(cox_baseline_hazard_death.index)[:-1] + [times.max()]
        cum_haz = [0] + list(cox_baseline_hazard_death.iloc[:, 0])

        start_time_cum_hazard = np.interp(list(y.time1), cum_haz_times, cum_haz)
        end_time_cum_hazard = np.interp(list(y.time2), cum_haz_times, cum_haz)
        cum_haz_diff = end_time_cum_hazard - start_time_cum_hazard
        survival_df = pd.DataFrame({'event': status, 'haz_diff': cum_haz_diff})
        return survival_df

    def ltrc_art_fit(self) -> DecisionTreeRegressor(criterion='poisson'):
        """
        The function fits an LTRCart tree with the given survival data.
        *Important Note*: The function uses cv in order to find the best ccp parameter to prune the tree, it is therefor
        time-consuming and difficult to use for random forest method without extensive parallel computing.
        :return: the function returns an @sklearn.tree.DecisionTreeRegressor with Poisson as its criterion, the
        regressor is fitted with the survival data entered after applying relevant data
        transformation(LTRCART.create_survival_data) and pruning the tree using best_ccp from a cv grid search.
        """
        new_survival_df = self.create_survival_data()
        est_survival_time = new_survival_df.event/new_survival_df.haz_diff
        tree_model = DecisionTreeRegressor(criterion="poisson", **self.control)

        tree_model.fit(X=self.x, y=est_survival_time, sample_weight=self.weights)
        # pruning the tree
        path = tree_model.cost_complexity_pruning_path(self.x, est_survival_time, self.weights)
        ccp_alphas, impurities = path.ccp_alphas, path.impurities
        # applying gridsearch to get the best ccp_alpha
        grid_search = GridSearchCV(DecisionTreeRegressor(criterion="poisson", **self.control),
                                   param_grid={'ccp_alpha': ccp_alphas}, scoring='neg_mean_poisson_deviance',
                                   n_jobs=3, cv=10, verbose=0, pre_dispatch='2*n_jobs', error_score=0)
        grid_search.fit(self.x, est_survival_time)
        best_ccp = grid_search.best_params_['ccp_alpha']
        if self.number_of_se_from_ccp == 0:
            clf = DecisionTreeRegressor(ccp_alpha=best_ccp, criterion="poisson", **self.control)
            clf.fit(X=self.x, y=est_survival_time, sample_weight=self.weights)
        else:
            best_score = grid_search.best_score_
            cv_results = grid_search.cv_results_
            cv_std = cv_results.loc[cv_results['param_ccp_alpha'] == best_ccp, 'std_test_score']

            se_from_best_score = best_score + cv_std * self.number_of_se_from_ccp
            ccp = cv_results.loc[cv_results.mean_test_score <= se_from_best_score, 'param_ccp_alpha'].max()
            clf = DecisionTreeRegressor(ccp_alpha=ccp, criterion="poisson", **self.control)
            clf.fit(X=self.x, y=est_survival_time, sample_weight=self.weights)
        return clf
