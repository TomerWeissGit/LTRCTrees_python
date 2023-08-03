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

        :param survival_data_obj:
        :param x:
        :param weights:
        :param number_of_se_from_ccp:
        :param control:
        """
        self.data = survival_data_obj.survival_data.copy()
        self.weights = weights
        self.number_of_se_from_ccp = number_of_se_from_ccp
        self.control = control
        self.x = x

    def create_survival_data(self) -> pd.DataFrame:
        y = self.data
        status = self.data.event.copy()
        times = self.data.time2.copy()
        unique_death_times = times[status == 1]
        cox_ph = CoxPHFitter()
        cox_baseline_cum_hazard = cox_ph.fit(self.data, 'time2', 'event').baseline_cumulative_hazard_
        # remove inf hazard and Nones.
        cox_baseline_cum_hazard = (cox_baseline_cum_hazard[~cox_baseline_cum_hazard
                                   .isin([np.nan, np.inf, -np.inf]).any(1)])
        cox_baseline_hazard_death = cox_baseline_cum_hazard.loc[cox_baseline_cum_hazard.index.isin(unique_death_times)]
        cum_haz_times = [0]+list(cox_baseline_hazard_death.index)[:-1]+[times.max()]
        cum_haz = [0] + list(cox_baseline_hazard_death.iloc[:, 0])

        start_time_cum_hazard = np.interp(cum_haz_times, cum_haz, [0]+list(y.time1))
        end_time_cum_hazard = np.interp(cum_haz_times, cum_haz, [0]+list(y.time2))
        new_time = end_time_cum_hazard-start_time_cum_hazard
        survival_df = pd.DataFrame({'event': status, 'time': new_time})
        return survival_df

    def ltrc_art_fit(self):
        tmp_survival_df = self.create_survival_data()
        tree_model = DecisionTreeRegressor(criterion="poisson", **self.control)
        tree_model.fit(X=self.x, y=tmp_survival_df, sample_weight=self.weights)
        # pruning the tree
        path = tree_model.cost_complexity_pruning_path(self.x, tmp_survival_df, self.weights)
        ccp_alphas, impurities = path.ccp_alphas, path.impurities
        # applying gridsearch to get the best ccp_alpha
        grid_search = GridSearchCV(DecisionTreeRegressor(criterion="poisson", **self.control),
                                   param_grid={'ccp_alpha': ccp_alphas}, scoring='r2', n_jobs=3,
                                   cv=10, verbose=0, pre_dispatch='2*n_jobs', error_score=0)
        best_ccp = grid_search.best_params_['ccp_alpha']
        if self.number_of_se_from_ccp == 0:
            clf = DecisionTreeRegressor(ccp_alpha=best_ccp)
            clf.fit(X=self.x, y=tmp_survival_df, sample_weight=self.weights)
        else:
            best_score = grid_search.best_score_
            cv_results = grid_search.cv_results_
            cv_std = cv_results.loc[cv_results['param_ccp_alpha'] == best_ccp, 'std_test_score']

            se_from_best_score = best_score+cv_std*self.number_of_se_from_ccp
            ccp = cv_results.loc[cv_results.mean_test_score <= se_from_best_score, 'param_ccp_alpha'].max()
            clf = DecisionTreeRegressor(ccp_alpha=ccp)
            clf.fit(X=self.x, y=tmp_survival_df, sample_weight=self.weights)
        return clf
