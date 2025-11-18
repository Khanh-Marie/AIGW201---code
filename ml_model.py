from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

class SurvivalPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.label_encoder = {}
        self.is_trained = False
        self.feature_columns = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "FamilySize", "IsAlone"]

    def preprocess_data(self, df, is_training=True):
        df_processed = df.copy()

        df_processed["Age"] = df_processed["Age"].fillna(df_processed["Age"].median())
        df_processed["Embarked"] = df_processed["Embarked"].fillna("S")
        df_processed["Fare"] = df_processed["Fare"].fillna(df_processed["Fare"].median())
        df_processed["FamilySize"] = df_processed["SibSp"] + df_processed["Parch"] + 1
        df_processed["IsAlone"] = (df_processed["FamilySize"] == 1).astype(int)

        categorical_cols = ["Sex", "Embarked"]
        for col in categorical_cols:
            if col not in self.label_encoder:
                self.label_encoder[col] = LabelEncoder()
                if is_training:
                    df_processed[col] = self.label_encoder[col].fit_transform(df_processed[col].astype(str))
                else:
                    df_processed[col] = self.label_encoder[col].transform(df_processed[col].astype(str))
            else:
                df_processed[col] = self.label_encoder[col].transform(df_processed[col].astype(str))

        x = df_processed[self.feature_columns]

        if is_training and "Survived" in df_processed.columns:
            y = df_processed["Survived"]
            return x, y
        else:
            return x, None

    def train(self, df):
        x, y = self.preprocess_data(df, is_training=True)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        x_train_scaled = self.scaler.fit_transform(x_train)
        x_test_scaled = self.scaler.transform(x_test)

        self.model.fit(x_train, y_train)
        y_pred = self.model.predict(x_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)

        self.is_trained = True
        return {
            "accuracy": accuracy,
            "classification_report": classification_report(y_test, y_pred, output_dict=True)
        }

    def predict_survival(self, passenger_data):
        if not self.is_trained:
            return {"error": "Not trained yet"}

        try:
            default_passenger = {
                "Pclass": 2,
                "Sex": "male",
                "Age": 30,
                "SibSp": 0,
                "Parch": 0,
                "Fare": 50,
                "Embarked": "S"
            }

            for key in passenger_data:
                if key in default_passenger:
                    default_passenger[key] = passenger_data[key]

            df_passenger = pd.DataFrame([default_passenger])
            x, _ = self.preprocess_data(df_passenger, is_training=False)
            x_scaled = self.scaler.transform(x)

            prediction = self.model.predict(x_scaled)[0]
            probability = self.model.predict_proba(x_scaled)[0]

            return {
                "survived" : bool(prediction),
                "probability_survived" : float(probability[1]),
                "probability_died": float(probability[0]),
                "passenger_info": default_passenger
            }
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}