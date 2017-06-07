# Success-Rate-Prediction

The dataset was from IBM watson project for predicting outcome of dog training. The dataset was extremely dirty with various missing values, different format in same feature, misspelled words and was divided into multiple files. All the features were cleaned using Python script 'clean_and_merge.py'. There were some fields that were beyond cleaning so I remvoed them. Since the data was divided into multiple files, I had to perform join on them to obtain the combined data. Combining data raised problems like repeated entries with for same dog with different values, so I had to make a judgement to which entry to keep and which to remove.

Section 1: Contains the dirty CSV files and the script to clean the data. outcome_inner_dog_left_person.csv is the file after performing cleaning script on it. The section also contains a correlation script to find features that are most relevant to the outcome of training.

Section 2: CrossValidationScript.py implements decision tree model on the cleaned data using cross validation for obtaining the best possible result. run10x.py performs prediction 10 times to check if the prediction are stable over the iterations.

Section 3: The dataset also had some text fields which contained long sentences from the trainer. Since models cant be trained on text, I converted these fields into features using various technologies like TF-IDF, Word2Vec, LDA and others. Then I used these features to implement the machine learning model.

Bonus: This section has details on how to use both the text features and the numerical features for better prediction.
