from lmstudio_client import LMStudioClient
from data_pipeline import DataAnalysisPipeline
from ml_model import SurvivalPredictor
import re


class AssistantManager:
    def __init__(self):
        self.data_pipeline = DataAnalysisPipeline()
        self.ml_model = SurvivalPredictor()
        self.lm_client = LMStudioClient()

        #Load and train on startup
        print("Loading Titanic dataset...")
        self.data_pipeline.load_titanic_data()
        print("Training ML model...")
        self.training_results = self.ml_model.train(self.data_pipeline.df)
        print("Assistant ready!")

    def detect_intent(self, user_message):
        message_lower = user_message.lower()

        if any(word in message_lower for word in ['analyze', 'analysis', 'data', 'statistics', 'titanic']):
            return "data_analysis"
        elif any(word in message_lower for word in ['predict', 'survival', 'survive', 'would survive']):
            return "prediction"
        elif any(word in message_lower for word in ['train', 'model', 'accuracy']):
            return "model_info"
        else:
            return "general_chat"

    def handle_data_analysis(self, user_message):
        try:
            analysis = self.data_pipeline.basic_analysis()
            visualizations = self.data_pipeline.generate_visualizations()

            #Safely access gender survival rates
            survival_rates = visualizations['survival_by_sex']
            female_rate = survival_rates.get('female', survival_rates.get('Female', 'N/A'))
            male_rate = survival_rates.get('male', survival_rates.get('Male', 'N/A'))

            summary = f"""
**📊 Titanic Dataset Analysis:**

**Dataset Overview:**
- Total passengers: {analysis['shape'][0]}
- Number of features: {analysis['shape'][1]}
- Overall survival rate: {visualizations['overall_survival']:.1%}

**Survival Analysis:**
- Female survival rate: {female_rate:.1%}
- Male survival rate: {male_rate:.1%}

**Passenger Demographics:**
- Average age: {visualizations['age_stats']['mean_age']:.1f} years
- Median age: {visualizations['age_stats']['median_age']:.1f} years

**Key Insights:**
The data shows significant differences in survival rates based on gender and class, with women and higher-class passengers having better survival chances.
"""
            return summary
        except Exception as e:
            return f"Error in data analysis: {str(e)}"

    def handle_prediction(self, user_message):
        try:
            #Extract passenger information from message
            passenger_data = {}

            #Extract class
            class_match = re.search(r'(\d)(?:st|nd|rd|th)\s*class', user_message.lower())
            if class_match:
                passenger_data['Pclass'] = int(class_match.group(1))
            else:
                class_match = re.search(r'class\s*(\d)', user_message.lower())
                if class_match:
                    passenger_data['Pclass'] = int(class_match.group(1))

            #Extract age
            age_match = re.search(r'(\d+)\s*year', user_message.lower())
            if age_match:
                passenger_data['Age'] = int(age_match.group(1))

            #Extract gender
            if 'female' in user_message.lower():
                passenger_data['Sex'] = 'female'
            elif 'male' in user_message.lower():
                passenger_data['Sex'] = 'male'

            #Extract fare if mentioned
            fare_match = re.search(r'fare\s*[\$]?\s*(\d+)', user_message.lower())
            if fare_match:
                passenger_data['Fare'] = float(fare_match.group(1))

            prediction = self.ml_model.predict_survival(passenger_data)

            if "error" in prediction:
                return f"❌ Prediction error: {prediction['error']}"

            survived_text = "✅ Yes" if prediction['survived'] else "❌ No"
            gender = passenger_data.get('Sex', 'unknown')
            age = passenger_data.get('Age', 'unknown')
            pclass = passenger_data.get('Pclass', 'unknown')

            result = f"""
**🔮 Survival Prediction:**

**Prediction:** {survived_text}
- Survival probability: {prediction['probability_survived']:.1%}
- Death probability: {prediction['probability_died']:.1%}

**Passenger Details:**
- Class: {pclass}
- Age: {age}
- Gender: {gender}

**Interpretation:**
This prediction is based on a machine learning model trained on historical Titanic passenger data.
"""
            return result
        except Exception as e:
            return f"Error in prediction: {str(e)}"

    def handle_model_info(self, user_message):
        return f"""
**🤖 ML Model Information:**

**Model Details:**
- Algorithm: Random Forest Classifier
- Training Accuracy: {self.training_results['accuracy']:.1%}
- Dataset: Titanic ({self.data_pipeline.df.shape[0]} passengers, {self.data_pipeline.df.shape[1]} features)

**Features Used:**
- Passenger class, gender, age, number of siblings/spouses, number of parents/children, fare, embarkation port, family size, and whether traveling alone

**Model Status:** ✅ Trained and ready for predictions
"""

    def process_message(self, user_message):
        intent = self.detect_intent(user_message)
        print(f"Detected intent: {intent} for message: {user_message}")

        if intent == "data_analysis":
            return self.handle_data_analysis(user_message)
        elif intent == "prediction":
            return self.handle_prediction(user_message)
        elif intent == "model_info":
            return self.handle_model_info(user_message)
        else:
            response = self.lm_client.chat_completion([{"role": "user", "content": user_message}])
            return response["choices"][0]["message"]["content"]
