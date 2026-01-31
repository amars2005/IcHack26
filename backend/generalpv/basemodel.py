import lightgbm as lgb

class BaseModel():
    def __init__(self, filepath, feature_names):
        self.filepath = filepath
        # Increased depth slightly to capture more complex spatial patterns
        self.model = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.03,
            num_leaves=63,
            max_depth=10,
            random_state=42,
            importance_type='gain'
        )
        self.feature_names = feature_names

    def train(self):
        

    