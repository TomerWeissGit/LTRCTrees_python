import sys

import pandas as pd
from dataclasses import dataclass


@dataclass
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
                self.survival_data = data.loc[['time1', 'time2', 'event']].copy()

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
