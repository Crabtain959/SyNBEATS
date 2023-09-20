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
    
    # self.dta: Pandas DataFrame with colnames = ['id', 'time', 'Y_obs']
    # treat_ids: List of Integers
    # target_time: Integer
    def __init__(self, dta, treat_ids, target_time, 
        date_format=None,
        input_size=1, output_size=1,
        ):
        
        self.date_format = date_format
        if self.date_format:
            # print(self.date_format)
            target_time = pd.to_datetime(target_time, format=self.date_format)
        # print(type(target_time))
        self.dta = dta
        self.treat_ids = treat_ids
        self.target_time = target_time
        self.input_size = input_size
        self.output_size = output_size

    
    def _prepare_data(self):
        if self.date_format:
            self.dta["time"] = pd.to_datetime(self.dta["time"], format=self.date_format)

        self.dta["id_tr"] = self.dta["id"].apply(lambda x: 1 if x in self.treat_ids else 0)
        self.dta["tr"] = np.where((self.dta["id"].isin(self.treat_ids)) & (self.dta["time"] >= self.target_time), 1, 0)

        all_control = list(set(self.dta['id']) - set(self.treat_ids))

        cov_list_all = [TimeSeries.from_dataframe(self.dta[self.dta['id'] == cid], 'time', 'Y_obs').astype(np.float32) for cid in all_control]
        cov_list_all = concatenate(cov_list_all, axis=1)

        ts_train_list = [TimeSeries.from_dataframe(self.dta[(self.dta['id'] == treat_id) & (self.dta['time'] < self.target_time)],
                                                'time', 'Y_obs') for treat_id in self.treat_ids]

        ts_list_train = concatenate(ts_train_list, axis=1)

        ts_test_list = [TimeSeries.from_dataframe(self.dta[(self.dta['id'] == treat_id) & (self.dta['time'] >= self.target_time)],
                                                'time', 'Y_obs') for treat_id in self.treat_ids]

        ts_list_test = concatenate(ts_test_list, axis=1)

        self.ts_list_test = ts_list_test
        self.ts_list_train = ts_list_train
        self.cov_list_all = cov_list_all

        
        

    def train(self, pred_length=-1, epochs=1500, lr=1e-4, batch_size=1024,
        patience=20, min_delta=0.005,
        use_gpu=None,
        verbose=True):

        early_stopping = EarlyStopping(
            monitor="train_loss", 
            patience=patience, 
            min_delta=min_delta
        )
        
        pl_trainer_kwargs = {"callbacks": [early_stopping], "accelerator": "gpu" if use_gpu else "cpu"}
        if use_gpu:
            pl_trainer_kwargs["devices"] = use_gpu

        the_model = NBEATSModel(
            input_chunk_length=self.input_size, 
            output_chunk_length=self.output_size, 
            n_epochs=epochs, 
            batch_size=batch_size, 
            optimizer_kwargs={'lr': lr}, 
            pl_trainer_kwargs=pl_trainer_kwargs
        )

        self._prepare_data()

        if pred_length == -1:
            pred_length = len(self.dta[(self.dta['time'] >= self.target_time) & (self.dta['id'] == self.treat_ids[0])])


        the_model.fit(series=self.ts_list_train, past_covariates=self.cov_list_all, verbose=verbose)
        self.darts_pred = the_model.predict(n=pred_length, series=self.ts_list_train, past_covariates=self.cov_list_all, verbose=verbose)
    
    def predictions(self, df=False):
        if df:
            return self.darts_pred.pd_dataframe()
        return self.darts_pred
    
    def backtest(self):
        ts_all_list = [TimeSeries.from_dataframe(self.dta[self.dta['id'] == treat_id], 'time', 'Y_obs') for treat_id in self.treat_ids]
        ts_list_all = concatenate(ts_all_list, axis=1)
        self.ts_list_all = ts_list_all

        backtest = the_model.historical_forecasts(series=ts_list_all, past_covariates=self.cov_list_all, retrain=False, 
                                                start=pd.to_datetime('1971', format="%Y"))

        backtest_df = backtest.pd_dataframe().reset_index()
        self.backtest = backtest
        backtest_df.columns = ["time"] + [f"prediction_{treat_id}" for treat_id in self.treat_ids]

    def plot_predictions(self, title="Prediction Plot", l_obs='Observed', l_pred='Predicted'):
        # forecasted_series = self.ts_list_train.append(self.darts_pred)
        true_series = self.ts_list_train.append(self.ts_list_test)
        true_series.plot(label=l_obs)
        self.darts_pred.plot(label=l_pred)
        
        plt.axvline(x=self.target_time, color='gray', linestyle='--', label='Target Time')

        
        plt.title(title)
        plt.legend()
        plt.show()

    def plot_backtest(self, title="Backcast Plot", l_obs='Observed', l_pred='Predicted'):
        self.backtest()
        self.ts_list_all.plot(label=l_obs)
        self.backtest.plot(label=l_pred)
        
        plt.title(title)
        plt.legend()
        plt.show()

    def plot_gap(self, l='Observed - Predicted', title="Gap Plot"):
        gap = self.ts_list_test - self.darts_pred
        gap.plot(label=l)

        plt.axhline(y=0, color='gray', linestyle='--')
        lim = max(abs(gap.values().flatten()))*1.05
        plt.ylim(-lim, lim)
        
        plt.title(title)
        plt.legend()
        plt.show()
        
    def _gap(self):
        return self.ts_list_test - self.darts_pred


    def average_treatment_effect(self):
        gap = self._gap()
        ate = gap.values().mean().item()
        return ate

    def std_treatment_effect(self):
        gap = self._gap()
        std = gap.values().std().item()
        return std
    

    def placebo_test(self, control_ids=None, use_gpu=None, plot=True):
        if control_ids is None:
            control_ids = list(set(self.dta['id']) - set(self.treat_ids))

        placebo_effects = {}

        print("Starting the placebo test...")

        pbar = tqdm(control_ids, desc='Processing placebo for control id')
        ates = []
        gaps = []
        for cid in pbar:
            pbar.set_description(f'Processing placebo for control id {cid}')
            placebo_model = SyNBEATS(self.dta, [cid], self.target_time, None,
                                    self.input_size, self.output_size)
            placebo_model.train(verbose=False, use_gpu=use_gpu)
            placebo_effects[cid] = placebo_model.predictions()
            ates.append(placebo_model.average_treatment_effect())
            gaps.append(placebo_model._gap())
        print("Placebo test completed.")
        
        if plot:
            plt.figure()
            
            lim = 0
            for gap in gaps:
                plt.plot(gap.time_index.tolist(), gap.values().squeeze(), color='gray', linewidth=0.5)
                lim = max(lim, max(abs(gap.values().squeeze()))*1.05)

            gap = self.ts_list_test - self.darts_pred
            plt.plot(gap.time_index.tolist(), gap.values().squeeze(), color='b', linewidth=2)
            
            lim = max(lim, max(abs(gap.values().squeeze()))*1.05)
            plt.axhline(y=0, color='gray', linestyle='--')
            plt.ylim(-lim, lim)

            plt.title('Placebo Effect Plot')
            plt.legend(['control', 'treat'])
            plt.show()

        
        actual_ate = self.average_treatment_effect()
        placebo_means = np.mean(ates)
        std_dev = np.std(ates)
        z_score = (actual_ate - placebo_means) / std_dev

        p_value = 1 - norm.cdf(abs(z_score))
        
        return placebo_effects, p_value

    def check_significance(self, placebo_effects, alpha=0.05):
        actual_ate = self.average_treatment_effect()
        placebo_means = np.mean(list(placebo_effects.values()))
        std_dev = np.std(list(placebo_effects.values()))
        z_score = (actual_ate - placebo_means) / std_dev

        p_value = 1 - norm.cdf(abs(z_score))
        critical_value = norm.ppf(1 - alpha/2)

        return abs(z_score) > critical_value, p_value

