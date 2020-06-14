# Disaster Response Pipeline Project
------------------------------------
![Image of message box](https://github.com/Vikram25/Udacity-Data-Science/blob/master/Disaster%20Response%20Project/screenshots/message%20box.png)

### Description:
----------------
This project is a part of Data Science Nano Degree program by Udacity where you will build a model for an API that classifies disaster messages.The initial dataset contains pre-labelled tweet and messages from real-life disaster.

The Project is divided into three parts:
    - Data Processing ETL Pipeline to extract data from source, clean data and save them in a proper databse structure
    - Machine Learning Pipeline to train a model able to classify text message in categories
    - Web App to show model results in real time.

### Project
-----------
### Dependencies

    - Python 3.6+
    - Machine Learning Libraries: NumPy, SciPy, Pandas, Sciki-Learn
    - Natural Language Process Libraries: NLTK
    - SQLlite Database Libraqries: SQLalchemy
    - Web App and Data Visualization: Flask, Plotly


### Instructions to run app:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Acknowledgements
[Udacity](Udacity.com) For providing industry level nanodegree program.
special thanks too [matteobonanomi](https://github.com/matteobonanomi)

### Screenshots
1. Example of messages you can type to test ML model
![Image of message](https://github.com/Vikram25/Udacity-Data-Science/blob/master/Disaster%20Response%20Project/screenshots/message.png)

2. After clicking classify Message. you will able to see the catagories which it belongs in highlighted green.
![Image of message box](https://github.com/Vikram25/Udacity-Data-Science/blob/master/Disaster%20Response%20Project/screenshots/result.png)

3. Graphs generated from training datasets
![Image of message box](https://github.com/Vikram25/Udacity-Data-Science/blob/master/Disaster%20Response%20Project/screenshots/plot1.png)
![Image of message box](https://github.com/Vikram25/Udacity-Data-Science/blob/master/Disaster%20Response%20Project/screenshots/plot2.png)
