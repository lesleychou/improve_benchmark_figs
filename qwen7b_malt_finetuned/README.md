# The fine-tuned model performance on App-MALT (datacenter capcaity planning)

## Run
```python
python plot_bars.py
```
X-axis represents 4 different fine-tuned model trained on different data: fine-tuned on only Level-1, only Level-2, only Level-3, and all levels data.

Y-axis represents the correctness and its confidence interval. Each color of bar is one type of testing data. 

E.g., blue bar in "Level-2" on x-axis means a model fine-tuned on Level-2 training data, testing on Level-1 data.



