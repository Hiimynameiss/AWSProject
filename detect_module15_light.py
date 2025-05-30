
import pandas as pd
from datetime import datetime, timedelta
import random

def get_anomaly_df():
    """
    테스트용 이상치 DataFrame 반환 (module15)
    """
    base_time = datetime(2025, 5, 1, 0, 0, 0)
    timestamps = [base_time + timedelta(minutes=10*i) for i in range(50)]
    total_errors = [round(random.uniform(0.1, 1.5), 3) for _ in range(50)]
    top_features = [random.choice(['current_S', 'activePower', 'powerfactor_R']) for _ in range(50)]

    df = pd.DataFrame({
        'timestamp': timestamps,
        'total_error': total_errors,
        'top_1_feature': top_features
    })
    return df
