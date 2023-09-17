# Class
`class SyNBEATS(dta, treat_ids, target_time, date_format=None, input_size=1, output_size=1)`

1. `dta`: DataFrame
input data with columns `['id', 'time', 'Y_obs']`. The values in 'time' column should be either in time format or evenly-spaced integers
2. `treat_ids`: list[int]
list of treated IDs
3. `target_time`: int
target time for the treatment
4. `date_format`: str, optional, default=`None`
format of the date in the 'time' column. `None` for integer values
5. `input_size`: int, optional, default=`1`
size of the input for the model
6. `output_size`: int, optional, default=`1`
size of the output for the model

# Methods
`_prepare_data()`
Prepares the data for training by creating several derived columns and setting up the time series data structures.

`train(pred_length=-1, epochs=1500, lr=1e-4, batch_size=1024, patience=20, min_delta=0.005, use_gpu=None, verbose=True)`
Trains the N-BEATS model using the prepared data.

1. `pred_length`: int, optional,  default=`-1`
prediction length, default is `-1` which calculates all the way back to `target_time`
2. `epochs`: int, optional, default=`1500`
number of training epochs
3. `lr`: float, optional, default=`1e-4`
learning rate,
4. `batch_size`: int, optional, default=`1024`
training batch size
5. `patience`: int, optional, default=`20`
patience for early stopping
6. `min_delta`: float, optional, default=`0.005`
minimum delta for early stopping
7. `use_gpu`: int, None, or a list of ints, optional, default=`'None'`
default is None which uses CPU, an integers sets the number of GPUs to be used, and a list of integers sets the indices of specific GPUs to be used
8. `verbose`: bool, optional, default=`True`
whether to print verbose training messages

`predictions()`
Returns the predictions after `target_time` as a pandas DataFrame.

`backtest()`
Backtests the model using historical data. Computationally intensive.

`plot_predictions(self, title="Prediction Plot", l_obs='Observed', l_pred='Predicted')`:
Plots the predictions along with the true values.
1. `title`: str, optional, default=`"Prediction Plot"`
title of the plot
2. `l_obs`: str, optional, default=`"Observed"`
label for observed data
3. `l_pred`: str, optional, default=`"Predicted"`
label for predicted data

`plot_backtest(self, title="Backcast Plot", l_obs='Observed', l_pred='Predicted')`
Plots the backtests along with the true values.
1. `title`: str, optional, default=`"Backtest Plot"`
title of the plot
2. `l_obs`: str, optional, default=`"Observed"`
label for observed data
3. `l_pred`: str, optional, default=`"Predicted"`
label for predicted data

`plot_gap(self, l='Observed - Predicted', title="Gap Plot")`
Plots the gap between the true values and the predictions.
1. `title`: str, optional, default=`"Gap Plot"`
title of the plot
2. `l`: str, optional, default=`"Observed - Predicted"`
label for the gap values

`average_treatment_effect()`
Calculates and returns the average treatment effect.

`std_treatment_effect()`
Calculates and returns the standard deviation of the treatment effect.

`placebo_test(control_ids=None, use_gpu=None, plot=True)`
Performs a placebo test using control groups, and returns a dictionary for the predictions for each contrl ID and a float for p-value of the placebo test.
1. `control_ids`: list[int] or `None`, optional, default=`None`
list of control IDs for the placebo test,  default is None which uses all non-treatment IDs
2. `use_gpu`: int, None, or a list of ints, optional, default=`'None'`
default is None which uses CPU, an integers sets the number of GPUs to be used, and a list of integers sets the indices of specific GPUs to be used
3. `plot`: bool, optional, default=`True`
whether to plot the predictions for the control IDs

# Example
**See example_usage.ipynb**

# Acknowledgements

This project is based on [Darts](https://github.com/unit8co/darts), which is licensed under the [Apache License 2.0](https://github.com/unit8co/darts/blob/develop/LICENSE). We are grateful to the creators and contributors of Darts for their work.
