<h1 align="center">SyNBEATS</h1>

## Description
This repo includes a [Python implementation](./SyNBEATS.py) of the SyNBEATS algorithm presented in this [paper](https://arxiv.org/abs/2208.03489). It also includes an [example usage notebook](./example_usage) and the [dataset](./smoking_data) used in it. 

## Installation
Download the [SyNBEATS.py](./SyNBEATS.py) and the [requirements.txt](./requirements.txt) and put it inside your working directory. \
Next run the following command to install the dependencies.
```bash
pip install -r requirements.txt
```
Then add the following line to your script to import the code. 
```python
from synbeats import *
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
model.plot_predictions()
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
class SyNBEATS(dta, treat_ids, target_time, date_format=None, input_size=1, output_size=1)
```
**_Parameters_**:
1. **`dta`**: Pandas DataFrame\
Input data with columns `['id', 'time', 'Y_obs']`. The values in 'time' column should be either in time format or evenly-spaced integers
2. **`treat_ids`**: list of integers\
List of IDs of treated units
3. **`target_time`**: integer\
Target time for the treatment
4. **`date_format`**: string, optional, default=`None`\
Format of the date in the 'time' column. `None` for integer values
5. **`input_size`**: integer, optional, default=`1`\
Size of the input for the model
6. **`output_size`**: integer, optional, default=`1`\
Size of the output for the model

### Methods
```python
train(pred_length=-1, epochs=1500, lr=1e-4, batch_size=1024, patience=20, min_delta=0.005, use_gpu=None, verbose=True)
```
Trains the N-BEATS model using the prepared data.

**_Parameters_**:
1. **`pred_length`**: int, optional,  default=`-1`\
Prediction length, default is `-1` which calculates all the way back to **`target_time`**
2. **`epochs`**: int, optional, default=`1500`\
Number of training epochs
3. **`lr`**: float, optional, default=`1e-4`\
Learning rate,
4. **`batch_size`**: int, optional, default=`1024`\
Training batch size
5. **`patience`**: int, optional, default=`20`\
Patience for early stopping
6. **`min_delta`**: float, optional, default=`0.005`\
Minimum delta for early stopping
7. **`use_gpu`**: int, None, or a list of ints, optional, default=`'None'`\
Default is None which uses CPU, an integers sets the number of GPUs to be used, and a list of integers sets the indices of specific GPUs to be used
8. **`verbose`**: bool, optional, default=`True`\
Whether to print verbose training messages

---

```python
predictions()
```
Returns the predictions after **`target_time`** as a pandas DataFrame.

---

```python
backtest()
```
Backtests the model using historical data. Computationally intensive.

---

```python
plot_predictions(self, title="Prediction Plot", l_obs='Observed', l_pred='Predicted')
```
Plots the predictions along with the true values.

**_Parameters_**:
1. **`title`**: str, optional, default=`"Prediction Plot"`\
Title of the plot
2. **`l_obs`**: str, optional, default=`"Observed"`\
Label for observed data
3. **`l_pred`**: str, optional, default=`"Predicted"`\
Label for predicted data

**_Example Plot_**:
<p align="center">
  <img src="https://github.com/Crabtain959/SyNBEATS/blob/f4923a163145e424f8152ce6e67793c88e48f0dc/plots/predictions.png">
</p>

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
Plots the gap between the true values and the predictions.

**_Parameters_**:
1. **`title`**: str, optional, default=`"Gap Plot"`\
Title of the plot
2. **`l`**: str, optional, default=`"Observed - Predicted"`\
Label for the gap values

**_Example plot_**:

<p align="center">
  <img src="https://github.com/Crabtain959/SyNBEATS/blob/f4923a163145e424f8152ce6e67793c88e48f0dc/plots/gap.png">
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
Performs a placebo test using control groups, and returns a dictionary for the predictions for each contrl ID and a float for p-value of the placebo test.

**_Parameters_**:
1. **`control_ids`**: list[int] or `None`, optional, default=`None`\
List of control IDs for the placebo test,  default is None which uses all non-treatment IDs
2. **`use_gpu`**: int, None, or a list of ints, optional, default=`'None'`\
Default is None which uses CPU, an integers sets the number of GPUs to be used, and a list of integers sets the indices of specific GPUs to be used
3. **`plot`**: bool, optional, default=`True`\
Whether to plot the predictions for the control IDs

**_Example Plot_**:
<p align="center">
  <img src="https://github.com/Crabtain959/SyNBEATS/blob/f4923a163145e424f8152ce6e67793c88e48f0dc/plots/placebo.png">
</p>

## Acknowledgements

This project is based on [Darts](https://github.com/unit8co/darts), which is licensed under the [Apache License 2.0](https://github.com/unit8co/darts/blob/develop/LICENSE). We are grateful to the creators and contributors of Darts for their work.
