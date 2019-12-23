# DSND_Disaster_Response_Pipelines
Analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages.

### Table of Contents

1. [Installation](#installation)
2. [Project Introduction](#introduction)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

The code should run with no issues using Python versions 3.*.
Libraries used in notebook/.py are:
- pandas
- sqlalchemy
- re
- nltk
- sklearn
- pickle
- plotly
- json
- flask


## Project Introduction<a name="introduction"></a>

In this project, I did following steps to build the model and use it in a web app:

1. Data Process: 

   - Loads the messages and categories datasets
   - Merges the two datasets
   - Cleans the data
   - Stores it in a SQLite database

2. Train classifier model: 

   - Loads data from the SQLite database
   - Splits the dataset into training and test sets
   - Builds a text processing and machine learning pipeline
   -  Trains and tunes a model using GridSearchCV
   -  Outputs results on the test set
   -  Exports the final model as a pickle file

3. Flask Web App:

   - Modify file paths for database and model as needed
   - Add data visualizations using Plotly in the web app. One example is provided for you


## File Descriptions <a name="files"></a>

**disaster_response_pipeline_project FOLDER**: all project deliever files are in this folder. 

**ETL Pipeline Preparation-zh.ipynb**: jupyter notebook to try and test ETL piplines. 

**ML Pipeline Preparation-zh.ipynb**: jupyter notebook to try and test ML piplines.  

**All other files**: they are data source file or DB/model files which generate by jupyter notebook. Just ignore those files I didn't mention above.


## Results<a name="results"></a>

The trained model is stored here: model/classifier.pkl. (however, due to the file is large, I cannot push to github)


## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Must give credit to Figure Eight for the data. This data is only use for Udacity Data Scientist Project. 

