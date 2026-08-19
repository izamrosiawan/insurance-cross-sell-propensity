import os
import joblib
import pandas as pd
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'cross_sell_model.joblib')

class CrossSellEngine:
    def __init__(self):
        saved = joblib.load(MODEL_PATH)
        self.model = saved['model']
        self.feature_names = saved['feature_names']

    def predict_propensity(self, df: pd.DataFrame) -> np.ndarray:
        df = df.copy()
        if 'Gender' in df.columns and df['Gender'].dtype == object:
            df['Gender'] = (df['Gender'] == 'Male').astype(int)
        if 'Vehicle_Damage' in df.columns and df['Vehicle_Damage'].dtype == object:
            df['Vehicle_Damage'] = (df['Vehicle_Damage'] == 'Yes').astype(int)
        if 'Vehicle_Age' in df.columns and df['Vehicle_Age'].dtype == object:
            vage_map = {'< 1 Year': 0, '1-2 Year': 1, '> 2 Years': 2}
            df['Vehicle_Age'] = df['Vehicle_Age'].map(vage_map)
            
        if 'Age_Vehicle_Interaction' not in df.columns and 'Age' in df.columns and 'Vehicle_Age' in df.columns:
            df['Age_Vehicle_Interaction'] = df['Age'] * (df['Vehicle_Age'] + 1)
        if 'Log_Annual_Premium' not in df.columns and 'Annual_Premium' in df.columns:
            df['Log_Annual_Premium'] = np.log1p(df['Annual_Premium'])
            
        df = df[self.feature_names]
        return self.model.predict_proba(df)[:, 1]
