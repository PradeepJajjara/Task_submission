# Task_submission
Task submission for Jatayu nlp engineer position
# Installing files
 - Install using "pip install -r requirements.txt"
# Dataset
 - Dataset using in training is in "data" folder
 - Positive sentiments are in pos_hindi and negative in neg_hindi
 - This dataset is gathered from github
 - Other dataset is in train,valid csv files
 - This dataset is from [Hindi_movies](https://www.kaggle.com/disisbig/hindi-movie-reviews-dataset) - this kaggle link.
# Training
 - Model training is showed in "Muril_training.ipynb"
 
# How to run
 ## Important 
  - Download muril file from [https://tfhub.dev/google/MuRIL/1] this link for running offline.
  - Else run normally
 To run type this in command console
  - uvicorn app:app --reload
