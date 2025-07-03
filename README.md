# HousingRegression

**Assignment 1: Boston Housing Price Prediction**

## Overview

* Compare Linear Regression, Random Forest, SVR against MSE & R².
* Extend with hyperparameter tuning via GridSearchCV.
* Automate via GitHub Actions.

## Repo Layout

```
.github/workflows/ci.yml   CI definitions
utils.py                   data & metrics
regression.py              regression models
requirements.txt           dependencies
```

## Quick Start

1. Create env & install:

   ```bash
   conda create -n venv
   conda activate venv
   pip install -r requirements.txt
   ```
2. Basic models (reg\_branch):

   ```bash
   git checkout reg_branch
   python regression.py
   ```
   
3. Tuned models (hyper\_branch):

```bash
   git checkout hyper_branch
   python regression.py
```

## CI Pipeline

* Runs on each push: installs deps, executes both scripts, fails on error.

