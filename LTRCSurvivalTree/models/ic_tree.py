from __future__ import annotations

import sys
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter


class SurvivalData(pd.DataFrame):

  def __init__(self, data: pd.DataFrame):
    """
    :param data: a pd.DataFrame with time1,time2,event columns or a pd.DataFrame with only three columns ordered by
    left truncation, right censoring and an event column.
    """
    pd.DataFrame.__init__(self)
    if data.shape[1] == 3:
      self.survival_data = data.copy()
      self.survival_data.columns = ['time1', 'time2', 'event']
    else:
      try:
        self.survival_data = data.loc[:, ['time1', 'time2', 'event']].copy()

      except LookupError as e:
        tb = sys.exception().__traceback__
        print("Response must be a 'pd.DataFrame' object with time1,time2,event columns.")
        raise e.with_traceback(tb)
    try:
      self.survival_data.loc[:, 'time1'] = self.survival_data.loc[:, 'time1'].astype(int)
      self.survival_data.loc[:, 'time2'] = self.survival_data.loc[:, 'time2'].astype(int)
    except ValueError as e:
      tb = sys.exception().__traceback__
      print("Response must be from a integer type columns.")
      raise e.with_traceback(tb)

class ICTree:

  def __init__(self,formula, survival_data: SurvivalData):
    self.data = survival_data.copy()

  def ic_tree(self):
    left = self.data.time1
    right = self.data.time2
    if sum(left == right)>=0:
      unique_times = list(set(list(left.values)+list(right.values)))
      unique_times.sort()
      lag_unique_times = np.array(unique_times[1:])
      lead_unique_times = np.array(unique_times[:-1])
      epsilon = np.min(lag_unique_times-lead_unique_times)/20
      self.data.loc[left==right,'time2'] += epsilon
    if self.data.time1.isin([np.inf, -np.inf]):
      raise ValueError("Infinity values are in left side of the interval, make sure well defined and bounded")
    if np.isinf(self.data.time2).values.sum() == self.data.time2.shape[0]:
      raise ValueError("all values in right side of interval are Infinity, unable to compute")
    # replace infinity values with max value*100
    self.data.time2[np.isinf(right).values] = right[~np.isinf(right).values].max()*100

    return partykit::ctree(formula=Formula, data=self.data, ytrafo=self.h2(weights=weights),
                           control=Control)
  ## x2 is Surv(Left,right,type="interval2") object
  @staticmethod
  def log_rank_transformation(survival_data: SurvivalData) -> pd.Series:
    """
    TODO: write more informative information about this function
    The function gets Interval censored survival data and returns log rank tranformation
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

  def h2(self, weights: None | list | np.array | pd.Series):
    weights = np.array([1] * self.data.shape[0]) if weights is None else np.array(weights)
    log_rank = self.log_rank_transformation(self.data.loc[weights > 0].copy())
    r = pd.Series([0]*len(weights))
    r[weights > 0] = log_rank
    list(estfun=matrix( float(r), ncol = 1), converged = TRUE)


