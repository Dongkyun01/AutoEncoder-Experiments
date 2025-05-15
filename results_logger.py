import json
import os

class ResultsLogger:
    def __init__(self, log_file="results.json"):
        self.log_file = log_file
        self.results = {}
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                self.results = json.load(f)

    def log(self, model_name, train_loss, test_loss):
        self.results[model_name] = {
            "train_loss": round(train_loss, 6),
            "test_loss": round(test_loss, 6)
        }
        self._save()

    def _save(self):
        with open(self.log_file, "w") as f:
            json.dump(self.results, f, indent=2)

    def report(self):
        print("📊 AutoEncoder Comparison Report")
        print("=" * 40)
        for model, scores in self.results.items():
            print(f"Model: {model}")
            print(f"  Train Loss: {scores['train_loss']:.6f}")
            print(f"  Test Loss : {scores['test_loss']:.6f}")
            print("-" * 40)

    def get_dataframe(self):
        import pandas as pd
        return pd.DataFrame(self.results).T
