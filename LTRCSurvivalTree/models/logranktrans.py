import sys
import numpy as np
from lifelines import KaplanMeierFitter
import pandas as pd


class SurvivalData:

    def __init__(self, data: pd.DataFrame):
        """
        :param data: a pd.DataFrame with time1,time2,event columns or a pd.DataFrame with only three columns ordered by
        left truncation, right censoring and an event column.
        """
        if data.shape[1] == 3:
            self.survival_data = data.copy()
        else:
            try:
                self.survival_data = data.loc[:, ['time1', 'time2', 'event']].copy()
            except LookupError as e:
                tb = sys.exception().__traceback__
                print("Response must be a 'pd.DataFrame' object with time1,time2,event columns.")
                raise e.with_traceback(tb)

    def log_rank_transformation(self):
        """

        :return:
        """

        # Fit IC survival curve
        kmf = KaplanMeierFitter()
        kmf.fit_interval_censoring(self.survival_data['time1'], self.survival_data['time2'])
        # get estimated survival
        left_predicted = 1 - kmf.cumulative_density_at_times(self.survival_data['time1'])
        right_predicted = 1 - kmf.cumulative_density_at_times(self.survival_data['time2'])

        log_left = 0 if left_predicted <= 0 else left_predicted * np.log(left_predicted)
        log_right = 0 if right_predicted <= 0 else right_predicted * np.log(right_predicted)
        result = (log_left - log_right) / (left_predicted - right_predicted)

        return result
