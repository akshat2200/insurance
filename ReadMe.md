# Machine Learning Project for Ultimate Claim

## Project: Ultimate Claim for Head On Collision Data using FNOL

## Install


This project requires **Python 3** with GPU Hardware accelerator(for enchanced performance) and the following Python libraries installed:


- [NumPy]
- [Pandas]
- [matplotlib]
- [scikit-learn]
- [seaborn]
- [xgboost]

Clone the environment_insurance.yml file before you start


## Data

Train data is given by __########__. I have split data into Train and Test in 8:2 ratio

## Code

Code is divided into two sections
 - Python Notebook (Under Code Tab) - Is used for data manipulation/cleaning/visulation and model selection
    -   insurance.ipynb NoteBook is used for data and model selection process
    -   model_pipeline_full.ipynb is used for training selected model and to create pipeline for future use
- Automation - Codes with pickle pipeline of trained model is used to run on future test or unseen data
    -   autoscript.py - check for new data in Input folder. If present then only run model
    -   predict.py - use pickle trained model pipeline to predict claim amount and saved with claim number in Output folder
    -   config.ini - can be used to change folder location easily for automation code


## Data Preprocessing and Visualization
Data visualization is done using seaborn heatmap and pandas scatter matrix to visualization correlation among features.
- [seaborn heatmaps](https://seaborn.pydata.org/generated/seaborn.heatmap.html)
- [pandas scatter matrix](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.plotting.scatter_matrix.html)  

Data Preprocessing is done using PCA to reduce dimensioanlity of data and transformation is done with PowerTransformer to achieve normal distribution.
- [Principal Component Analysis](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
- [Power Transformer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html)


## Implementation
Three ML approaches discussed and implemented in this project: 
- [DecisionTreeRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)
- [RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
- [XGBRegressor](https://xgboost.readthedocs.io/en/latest/python/python_api.html)


## Model Evaluation and Validation
Model is tested on 20% of train data splitted from orginial train data in ratio of 8:2. Remaining 80% of train data is used for 
model training and validation.Metric used to measure performance of model is Mean Absolute error (MAE). Goal of all selected ML model approches implemented is to reduce MAE. MAE is average over difference between predicted and actual Incurred on test data.

## Results
MAE score from all three implementations are reported and final model is chosen with best MAE score. 