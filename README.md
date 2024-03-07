<h1 align="center">SyNBEATS</h1>

## Description
This repo includes a [Python implementation](./SyNBEATS.py) of the SyNBEATS algorithm presented in the paper: [Forecasting Algorithms for Causal Inference with Panel Data
](https://arxiv.org/abs/2208.03489). It also includes an [example usage notebook](./example_usage.ipynb) and the [dataset](./smoking_data.csv) used in it. The dataset originally comes from the paper [Adadie et al.](https://web.stanford.edu/~jhain/Paper/JASA2010.pdf)
## Installation
```bash
pip install git+https://github.com/Crabtain959/SyNBEATS.git
```
Or you can download this repo and run 
```python
pip install .
```


## Example Usage

```python
# Read the data into dta

# Import the class
from SyNBEATS import *

# Build and train the model
model = SyNBEATS(dta, [3], 1989)
model.train(use_gpu=1)

# Plot the predictions and the gap
predictions = model.predict()
model.plot_predictions(predictions)
model.plot_gap()

# Calculate the average treatment effect and the standard deviation of the treatment effects
ate = model.average_treatment_effect()
std = model.std_treatment_effect()

# Placebo Test
placebo_predictions, p_value = model.placebo_test(control_ids=[i for i in range(4,15)], use_gpu=1)
```
**See more in [example_usage.ipynb](./example_usage.ipynb)**

## Detailed Documentation

### Class
```python
class SyNBEATS(dta, treat_ids, target_time, control_ids=None, date_format=None, input_size=1, output_size=1)
```
**_Parameters_**:
1. **`dta`**: Pandas DataFrame\
Input data with columns `['id', 'time', 'Y_obs']`. The values in 'time' column should be either in time format or evenly-spaced integers
2. **`treat_ids`**: list\
List of IDs of treated units
3. **`target_time`**: integer\
Target time for the treatment
4. **`control_ids`**: list, optional, default=`None`\
List of IDs of control units. Default is `None` which uses all the other units as control units
5. **`date_format`**: string, optional, default=`None`\
Format of the date in the 'time' column. `None` for integer values
6. **`input_size`**: integer, optional, default=`1`\
Size of the input for the model
7. **`output_size`**: integer, optional, default=`1`\
Size of the output for the model

### Attributes
| **Name**      | **Description**                                                                                                          |
|---------------|--------------------------------------------------------------------------------------------------------------------------|
| date_format   | Format of the date in the 'time' column                                                                                  |
| dta           | input data                                                                                                               |
| treat_ids     | List of IDs of treated units                                                                                             |
| control_ids   | List of IDs of control units                                                                                             |
| target_time   | Target time for the treatment                                                                                            |
| input_size    | Size of the input for the model                                                                                          |
| output_size   | Size of the output for the model                                                                                         |
| ts_list_all   | Data corresponds to the treated units of all time                                                                        |
| ts_list_test  | Data corresponds to the treated units since target time                                                                  |
| ts_list_train | Data corresponds to the treated units before target time                                                                 |
| cov_list_all  | Data corresponds to the control units of all time                                                                        |
| model         | The trained model (initialized after calling `train()`)                                                                  |
| backtest      | Predictions that the model would have produced on the entire series without retrain  (initialized after calling `train`) |

### Methods
```python
train(epochs=1500, lr=1e-4, batch_size=1024, patience=20, min_delta=0.005, use_gpu=None, verbose=True)
```
Trains the N-BEATS model using the prepared data.

**_Parameters_**:
1. **`epochs`**: int, optional, default=`1500`\
Number of training epochs
2. **`lr`**: float, optional, default=`1e-4`\
Learning rate,
3. **`batch_size`**: int, optional, default=`1024`\
Training batch size
4. **`patience`**: int, optional, default=`20`\
Patience for early stopping
5. **`min_delta`**: float, optional, default=`0.005`\
Minimum delta for early stopping
6. **`use_gpu`**: int, None, or a list of ints, optional, default=`'None'`\
Default is None which uses CPU, an integers sets the number of GPUs to be used, and a list of integers sets the indices of specific GPUs to be used
7. **`verbose`**: bool, optional, default=`True`\
Whether to print verbose training messages

---

```python
predictions(pred_length=-1, df=False, verbose=True)
```
Returns the predictions after **`target_time`**.

**_Parameters_**:
1. **`pred_length`**: int, optional,  default=`-1`\
Prediction length, default is `-1` which calculates all the way back to **`target_time`**
2. **`df`**: boolean, optional, default=`False`\
Whether to return the predictions as a Pandas DataFrame. Default is `False` which returns a Darts TimeSeries instead. 
3. **`verbose`**: bool, optional, default=`True`\
Whether to print verbose training messages

---

```python
plot_predictions(self, predictions, title="Prediction Plot", l_obs='Observed', l_pred='Predicted')
```
Plots the predictions along with the true values. If the model is fitted on multiple units, it plots the mean of the predictions of all treated units along with the mean of the true values of them.

**_Parameters_**:
1. **`predictions`**: Darts TimeSeries
The predictions of the model
2. **`predict_pretreatment`**: boolean, optional, default=`"False"`\
Wether to plot the pre-treatment predictions. If not (set to `"False"`), only post-treatment predictions will be plotted.
4. **`title`**: str, optional, default=`"Prediction Plot"`\
Title of the plot
5. **`l_obs`**: str, optional, default=`"Observed"`\
Label for observed data
6. **`l_pred`**: str, optional, default=`"Predicted"`\
Label for predicted data

**_Example Plot_**:
<p align="center">
  <img src="./plots/predictions_without_pre.png">
</p>

<p align="center">
  <img src="./plots/predictions_with_pre.png">
</p>

---

```python
backtest()
```
Backtests the model using historical data. Computationally intensive.

---

```python
plot_backtest(self, title="Backcast Plot", l_obs='Observed', l_pred='Predicted')
```
Plots the backtests along with the true values.

**_Parameters_**:
1. **`title`**: str, optional, default=`"Backtest Plot"`\
Title of the plot
2. **`l_obs`**: str, optional, default=`"Observed"`\
Label for observed data
3. **`l_pred`**: str, optional, default=`"Predicted"`\
Label for predicted data

---

```python
plot_gap(self, l='Observed - Predicted', title="Gap Plot")
```
Plots the gap between the true values and the predictions. If the model is fitted on multiple treated units, it plots the mean of the gaps of all the treated units.

**_Parameters_**:
1. **`title`**: str, optional, default=`"Gap Plot"`\
Title of the plot
2. **`l`**: str, optional, default=`"Observed - Predicted"`\
Label for the gap values

**_Example plot_**:

<p align="center">
  <img src="./plots/gap.png">
</p>

---

```python
average_treatment_effect()
```
Calculates and returns the average treatment effect.

---

```python
std_treatment_effect()
```
Calculates and returns the standard deviation of the treatment effect.

---

```python
placebo_test(control_ids=None, use_gpu=None, plot=True)
```
Performs a placebo test using control groups, and returns a dictionary for the predictions for each contrl ID and a float for p-value of the placebo test. Multiple treated unit placebo test is not supported because it invloves too much intensive computation.

**_Parameters_**:
1. **`control_ids`**: list[int] or `None`, optional, default=`None`\
List of control IDs for the placebo test,  default is None which uses all non-treatment IDs
2. **`use_gpu`**: int, None, or a list of ints, optional, default=`'None'`\
Default is None which uses CPU, an integers sets the number of GPUs to be used, and a list of integers sets the indices of specific GPUs to be used
3. **`plot`**: bool, optional, default=`True`\
Whether to plot the predictions for the control IDs

**_Example Plot_**:
<p align="center">
  <img src="./plots/placebo.png">
</p>

## Acknowledgements

This project is based on [Darts](https://github.com/unit8co/darts), which is licensed under the [Apache License 2.0](https://github.com/unit8co/darts/blob/develop/LICENSE). We are grateful to the creators and contributors of Darts for their work.
