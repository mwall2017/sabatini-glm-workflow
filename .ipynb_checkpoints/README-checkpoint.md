sglm
==============================

A GLM Pipeline for Neuroscience Analyses

Built using sklearn.ElasticNet,Ridge, ElasticNetCV, and RidgeCV

All necessary packages listed in requirements.txt and are pip installable!

There are two notebooks within this repository: gridSearch and fitGLM.
* The gridSearch notebook is used to find the best parameters and will help you select the best regression model for your data.

* The fitGLM notebook is used to fit the model for known parameters. 

Both notebooks are similar in structure and will output a project directory with the necessary files to continue your analysis.


## Project Organization

The notebooks will output a project directory with the following structure:

```bash
Project directory will be created to include: 
|
| Project_Name
| ├── data
|   ├── 00001.csv
|   ├── 00002.csv
|   └── combined_output.csv
| ├── models
|   └── project_name.pkl
| ├── results
|   ├── model_fit.png
|   ├── predicted_vs_actual.png
|   └── residuals.png
| config.yaml
```

data folder: will include all of your data in .csv format. Please refer to the notebook for formatting.

models folder: will include outputs saved from the model_dict.

results folder: includes some figures for quick visualization. 

config.yaml: your config file to set your parameters for the model. 

## Troubleshooting:

* This is a work in progress and will be updated as needed. Please feel free to reach out with any questions or concerns.
