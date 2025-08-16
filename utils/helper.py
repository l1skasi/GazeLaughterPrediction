from sklearn.metrics import precision_score, f1_score
import numpy as np

def get_quality_metrics(preds, targets):
      accuracy = (np.array(preds) == np.array(targets)).mean()
      precision = precision_score(targets, preds, average='weighted', zero_division=0)
      f1 = f1_score(targets, preds, average='weighted', zero_division=0)
      return accuracy, precision, f1

