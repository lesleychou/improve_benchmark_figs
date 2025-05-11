# Script to analyze App-Route results

It will generate 3 spider charts, and two SEM (95% confidence interval) charts (one with small query number, one with large query number)

## Required packages (common matplotlib ones)
```
import os
import json
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import matplotlib.patches as mpatches
```

## Run
```python
python file_utils.py
```