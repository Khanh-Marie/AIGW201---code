import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


class DataAnalysisPipeline:
    def __init__(self):
        self.df = None
        self.analysis_results = {}

    def load_titanic_data(self):
        url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
        self.df = pd.read_csv(url)
        return self.df

    def basic_analysis(self):
        if self.df is None:
            self.load_titanic_data()

        analysis = {
            "shape": self.df.shape,
            "columns": list(self.df.columns),
            "data_types": self.df.dtypes.to_dict(),
            "missing_values": self.df.isnull().sum().to_dict(),
            "basic_stats": self.df.describe().to_dict()
        }
        self.analysis_results["basic"] = analysis
        return analysis

    def generate_visualizations(self):
        if self.df is None:
            self.load_titanic_data()

        #Fix: Handle case sensitivity in gender values
        survival_by_sex = self.df.groupby('Sex')['Survived'].mean()

        #Convert to lowercase keys for consistency
        survival_dict = {}
        for gender in survival_by_sex.index:
            survival_dict[gender.lower()] = survival_by_sex[gender]

        age_stats = {
            "mean_age": self.df['Age'].mean(),
            "median_age": self.df['Age'].median(),
            "age_std": self.df['Age'].std()
        }

        #Calculate overall survival rate
        overall_survival = self.df['Survived'].mean()

        self.analysis_results["visualizations"] = {
            "survival_by_sex": survival_dict,
            "age_stats": age_stats,
            "overall_survival": overall_survival
        }

        return self.analysis_results["visualizations"]
