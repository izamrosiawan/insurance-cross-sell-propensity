import pytest
import pandas as pd
import numpy as np
from src.cross_sell_engine import CrossSellEngine

def test_cross_sell_propensity_range():
    engine = CrossSellEngine()
    sample = pd.read_csv('data/train.csv', nrows=5)
    probs = engine.predict_propensity(sample)
    
    assert len(probs) == 5
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)
