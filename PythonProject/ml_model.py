from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np


class SurvivalPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.is_trained = False
        self.feature_columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'FamilySize', 'IsAlone']

    def preprocess_data(self, df, is_training=True):
        #Create a copy to avoid modifying original
        df_processed = df.copy()

        #Handle missing values
        df_processed['Age'] = df_processed['Age'].fillna(df_processed['Age'].median())
        df_processed['Embarked'] = df_processed['Embarked'].fillna('S')
        df_processed['Fare'] = df_processed['Fare'].fillna(df_processed['Fare'].median())

        #Feature engineering
        df_processed['FamilySize'] = df_processed['SibSp'] + df_processed['Parch'] + 1
        df_processed['IsAlone'] = (df_processed['FamilySize'] == 1).astype(int)

        #Encode categorical variables
        categorical_cols = ['Sex', 'Embarked']
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                if is_training:
                    #Fit on training data
                    df_processed[col] = self.label_encoders[col].fit_transform(df_processed[col].astype(str))
                else:
                    #For prediction, use existing encoder
                    df_processed[col] = self.label_encoders[col].transform(df_processed[col].astype(str))
            else:
                #Use existing encoder
                df_processed[col] = self.label_encoders[col].transform(df_processed[col].astype(str))

        #Select features
        X = df_processed[self.feature_columns]

        if is_training and 'Survived' in df_processed.columns:
            y = df_processed['Survived']
            return X, y
        else:
            return X, None

    def train(self, df):
        X, y = self.preprocess_data(df, is_training=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.model.fit(X_train_scaled, y_train)

        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)

        self.is_trained = True
        return {
            "accuracy": accuracy,
            "classification_report": classification_report(y_test, y_pred, output_dict=True)
        }

    def predict_survival(self, passenger_data):
        if not self.is_trained:
            return {"error": "Model not trained yet"}

        try:
            #Ensure all required features are present
            default_passenger = {
                'Pclass': 2,
                'Sex': 'male',
                'Age': 30,
                'SibSp': 0,
                'Parch': 0,
                'Fare': 50,
                'Embarked': 'S'
            }

            #Update with provided data
            for key in passenger_data:
                if key in default_passenger:
                    default_passenger[key] = passenger_data[key]

            df_passenger = pd.DataFrame([default_passenger])
            X, _ = self.preprocess_data(df_passenger, is_training=False)
            X_scaled = self.scaler.transform(X)

            prediction = self.model.predict(X_scaled)[0]
            probability = self.model.predict_proba(X_scaled)[0]

            return {
                "survived": bool(prediction),
                "probability_survived": float(probability[1]),
                "probability_died": float(probability[0]),
                "passenger_info": default_passenger
            }
        except Exception as e:

            return {"error": f"Prediction failed: {str(e)}"}
