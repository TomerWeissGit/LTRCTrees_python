from __future__ import annotations

from typing import Optional

import lifelines
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
        # The tree model which will be fitted
        self.tree = None

    def create_ltrcart_data(self) -> pd.DataFrame:
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
        poisson_df = pd.DataFrame({'event': status, 'haz_diff': cum_haz_diff})
        return poisson_df

    def fit(self) -> DecisionTreeRegressor(criterion='poisson'):
        """
        The function fits an LTRCart tree with the given survival data.
        *Important Note*: The function uses cv in order to find the best ccp parameter to prune the tree, it is therefor
        time-consuming and difficult to use for random forest method without extensive parallel computing.
        :return: the function returns an @sklearn.tree.DecisionTreeRegressor with Poisson as its criterion, the
        regressor is fitted with the survival data entered after applying relevant data
        transformation(LTRCART.create_survival_data) and pruning the tree using best_ccp from a cv grid search.
        """
        poisson_df = self.create_ltrcart_data()
        est_hazard = poisson_df.haz_diff

        tree_model = DecisionTreeRegressor(criterion="poisson", **self.control)

        tree_model.fit(X=self.x, y=est_hazard, sample_weight=self.weights)
        # pruning the tree

        path = tree_model.cost_complexity_pruning_path(self.x, est_hazard, self.weights)
        ccp_alphas, impurities = path.ccp_alphas, path.impurities

        # applying gridsearch to get the best ccp_alpha

        grid_search = GridSearchCV(DecisionTreeRegressor(criterion="poisson", **self.control),
                                   param_grid={'ccp_alpha': ccp_alphas}, scoring='neg_mean_poisson_deviance',
                                   n_jobs=3, cv=10, verbose=0, pre_dispatch='2*n_jobs', error_score=0)
        grid_search.fit(self.x, est_hazard)
        best_ccp = grid_search.best_params_['ccp_alpha']
        if self.number_of_se_from_ccp == 0:
            clf = DecisionTreeRegressor(ccp_alpha=best_ccp, criterion="poisson", **self.control)
            clf.fit(X=self.x, y=est_hazard, sample_weight=self.weights)
        else:
            best_score = grid_search.best_score_
            cv_results = grid_search.cv_results_
            cv_std = cv_results.loc[cv_results['param_ccp_alpha'] == best_ccp, 'std_test_score']

            se_from_best_score = best_score + cv_std * self.number_of_se_from_ccp
            ccp = cv_results.loc[cv_results.mean_test_score <= se_from_best_score, 'param_ccp_alpha'].max()
            clf = DecisionTreeRegressor(ccp_alpha=ccp, criterion="poisson", **self.control)
            clf.fit(X=self.x, y=est_hazard, sample_weight=self.weights)
        self.tree = clf
        return clf

    def predict(self, x_test: pd.DataFrame) -> dict:
        """
        The function predicts the km curve using the LTRCART algorithm, it simply builds the tree and uses the predicted
        RR as an id, more accurately, it groups the train data by the node of the tree which the train obs got to, then,
        it fits km module on this subgroup and assign it to the test observations reaching the same node.
        :param x_test: DataFrame with the exact fitted variables entered the init.
        :return: dictionary containing km curves and median survival time for test observations (predictions)
        """
        if self.tree is None:
            self.fit()

        self.x['id_rr'] = self.tree.predict(self.x)
        key = self.x.id_rr.unique
        keys_df = pd.DataFrame({'key': key, 'keys_id': range(len(key))})

        list_km = []
        list_med = []

        for p in key:
            subset = self.x.loc[self.x.id_rr == p]
            y = self.data.copy()
            y = y.loc[subset.index]
            km_fitter = lifelines.KaplanMeierFitter()
            km_fitter.fit(duration=y.time2, event_observed=y.event, entry=y.time1)
            sub_group_med_survival_time = km_fitter.median_survival_time_
            key_id = keys_df.loc[keys_df.key == p, 'keys_id']
            list_km[key_id] = km_fitter
            list_med[key_id] = sub_group_med_survival_time
        test = x_test.copy()
        test['key'] = self.tree.predict(x_test)
        test['keys_id'] = test.key.map(lambda x: keys_df.loc[keys_df.key == x, 'keys_id'])
        test_km = list_km[test.keys_id.values]
        test_med = list_med[test.keys_id.values]
        return {'km_curves': test_km, 'medians': test_med}
