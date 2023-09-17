import torch
import pandas as pd
from darts import TimeSeries, concatenate
from darts.models import NBEATSModel
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import logging
import numpy as np
from tqdm import tqdm
from scipy.stats import norm

logging.getLogger("pytorch_lightning.utilities.rank_zero").setLevel(logging.WARNING)
logging.getLogger("pytorch_lightning.accelerators.cuda").setLevel(logging.WARNING)

class SyNBEATS:
    # self.dta: Pandas DataFrame with colnames = ['id', 'year', 'Y_obs']
    # treat_ids: List of Integers
    # target_year: Integer
    def __init__(self, dta, treat_ids, target_year, 
        input_size=1, output_size=1,
        ):

        target_year = pd.to_datetime(target_year, format="%Y")

        self.dta = dta
        self.treat_ids = treat_ids
        self.target_year = target_year
        self.input_size = input_size
        self.output_size = output_size

        

    def forecast(self, pred_length=-1, epochs=1500, lr=1e-4, batch_size=1024,
        patience=20, min_delta=0.005,
        verbose=True):

        early_stopping = EarlyStopping(
            monitor="train_loss", 
            patience=patience, 
            min_delta=min_delta
            )

        the_model = NBEATSModel(
            input_chunk_length=self.input_size, 
            output_chunk_length=self.output_size, 
            n_epochs=epochs, 
            batch_size=batch_size, 
            optimizer_kwargs={'lr': lr}, 
            pl_trainer_kwargs={"callbacks": [early_stopping]}
            )

        self.dta["id_tr"] = self.dta["id"].apply(lambda x: 1 if x in self.treat_ids else 0)
        self.dta["tr"] = np.where((self.dta["id"].isin(self.treat_ids)) & (self.dta["year"] >= self.target_year), 1, 0)

        all_control = list(set(self.dta['id']) - set(self.treat_ids))

        cov_list_all = [TimeSeries.from_dataframe(self.dta[self.dta['id'] == cid], 'year', 'Y_obs').astype(np.float32) for cid in all_control]
        cov_list_all = concatenate(cov_list_all, axis=1)

        ts_train_list = [TimeSeries.from_dataframe(self.dta[(self.dta['id'] == treat_id) & (self.dta['year'] < pd.to_datetime(self.target_year, format="%Y"))],
                                                'year', 'Y_obs') for treat_id in self.treat_ids]

        ts_list_train = concatenate(ts_train_list, axis=1)

        ts_test_list = [TimeSeries.from_dataframe(self.dta[(self.dta['id'] == treat_id) & (self.dta['year'] >= pd.to_datetime(self.target_year, format="%Y"))],
                                                'year', 'Y_obs') for treat_id in self.treat_ids]

        ts_list_test = concatenate(ts_test_list, axis=1)

        self.ts_list_test = ts_list_test
        self.ts_list_train = ts_list_train

        if pred_length == -1:
            representative_id = self.treat_ids[0] 
            pred_length = len(self.dta[(self.dta['year'] >= self.target_year) & (self.dta['id'] == representative_id)])



        the_model.fit(series=ts_list_train, past_covariates=cov_list_all, verbose=verbose)
        self.darts_pred = the_model.predict(n=pred_length, series=ts_list_train, past_covariates=cov_list_all, verbose=verbose)

        ts_all_list = [TimeSeries.from_dataframe(self.dta[self.dta['id'] == treat_id], 'year', 'Y_obs') for treat_id in self.treat_ids]
        ts_list_all = concatenate(ts_all_list, axis=1)
        self.ts_list_all = ts_list_all

        backtest = the_model.historical_forecasts(series=ts_list_all, past_covariates=cov_list_all, retrain=False, 
                                                start=pd.to_datetime('1971', format="%Y"))

        backtest_df = backtest.pd_dataframe().reset_index()
        self.backtest = backtest
        backtest_df.columns = ["year"] + [f"prediction_{treat_id}" for treat_id in self.treat_ids]

        self.backtest_df = backtest_df

        return self.darts_pred

    def plot_forecast(self, title="Forecast Plot", l_obs='obs', l_pred='pred'):
        # forecasted_series = self.ts_list_train.append(self.darts_pred)
        true_series = self.ts_list_train.append(self.ts_list_test)
        true_series.plot(label=l_obs)
        self.darts_pred.plot(label=l_pred)
        
        plt.title(title)
        plt.legend()
        plt.show()

    def plot_backcast(self, title="Backcast Plot", l_obs='obs', l_pred='pred'):
        self.ts_list_all.plot(label=l_obs)
        self.backtest.plot(label=l_pred)
        
        plt.title(title)
        plt.legend()
        plt.show()

    def plot_gap(self, l='obs-pred', title="Gap Plot", color="b", shape="."):
        gap = self.ts_list_test - self.darts_pred
        gap.plot(color=color, marker=shape, label=l)
        
        plt.title(title)
        plt.legend()
        plt.show()

    def average_treatment_effect(self):
        gap = self.ts_list_test - self.darts_pred
        ate = gap.values().mean().item()
        return ate

    def std_treatment_effect(self):
        gap = self.ts_list_test - self.darts_pred
        std = gap.values().std().item()
        return std
    

    def placebo_test(self, control_ids=None):
        if control_ids is None:
            control_ids = list(set(self.dta['id']) - set(self.treat_ids))

        placebo_effects = {}

        print("Starting the placebo test...")

        pbar = tqdm(control_ids, desc='Processing placebo for control id')
        for cid in pbar:
            pbar.set_description(f'Processing placebo for control id {cid}')
            placebo_model = SyNBEATS(self.dta, [cid], self.target_year, 
                                    self.input_size, self.output_size)
            placebo_model.forecast(verbose=False)
            placebo_effects[cid] = placebo_model.average_treatment_effect()

        print("Placebo test completed.")
        return placebo_effects

    def check_significance(self, placebo_effects, alpha=0.05):
        actual_ate = self.average_treatment_effect()
        placebo_means = np.mean(list(placebo_effects.values()))
        std_dev = np.std(list(placebo_effects.values()))
        z_score = (actual_ate - placebo_means) / std_dev

        p_value = 1 - norm.cdf(abs(z_score))
        critical_value = norm.ppf(1 - alpha/2)

        return abs(z_score) > critical_value, p_value

