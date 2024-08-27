# Stroke Prediction using a Random Forest Classifier and Data Pipeline 

# Full data pipeline using preprocessing methods, feature engineering methods (statistically significant feature selector, zero variance threshold, and embedded ML feature selector), and a Random Forest Classifier as the predictor (homogeneous ensemble with fine-tuning) in order to determine if a patient has stroke given selected features. 

# EDA & Preprocessing: 
This project starts by using exploratory data analysis with Seaborn and Pandas to determine which preprocessing methods would be most effective to improve model accuracy. For continuous features, I used the standard scaler from scikit learn to prevent bias. For categorical features, I used Seaborn countplots to examine relationships between categorical features and stroke results (target feature). Using these plots, I determined some categorical features (gender, type of work, type of residence, and marriage status) didn't seem to have order among classes in terms of their relationship to patients with strokes. Therefore, I used one hot encoding to encode the features to feed into the predictor. For features I determined that had order in terms of relationship to the diagnosis of stroke, I used ordinal encoding to keep the order in which they're correlated intact. 

# Feature Engineering: 
I used a zero variance filter to remove features with little variance as their impact on stroke prediction is negligible. Then I used a feature union to apply a statistical feature selector (SelectKBest) which measures how statistically significant features are in relation to the target feature as well as an embedded random forest selector to then combine the features with the highest feature importances. 

# Model & Finetuning: 
This project utilizes a random forest ensemble to classify if patients have had a stroke or not. When training the model, I used GridSearchCV to finetune the parameters of the random forest classifier, the number of features that should be picked with the highest feature imporances, feature selector thresholds, model depth, and the number of decision trees in the ensemble. Using KFold split with 10 folds (statistically significant number of folds), I trained the model and tested it to achieve a 96% accuracy on unseen validation data. 

