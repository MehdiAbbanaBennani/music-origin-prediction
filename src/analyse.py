import json
import numpy as np
LOG_FILE = "logs/run_logs.json"

with open(LOG_FILE) as f:
  data = json.load(f)

min_loss_idx = np.argmin(data["test_acc"])

for key, val in data.items():
  print("{} {}", key, val[min_loss_idx])