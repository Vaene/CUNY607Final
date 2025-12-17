
# After filling in evaluation_template.csv, run this:

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

# Load evaluated data
df = pd.read_csv("output/evaluation_template.csv")

# Filter completed evaluations
df_complete = df[df['human_label'].notna() & (df['human_label'] != '')]

# Calculate accuracy
accuracy = (df_complete['correct'] == 'Yes').mean()
partial = (df_complete['correct'] == 'Partial').mean()

print(f"Exact Match Accuracy: {accuracy:.2%}")
print(f"Partial Match: {partial:.2%}")
print(f"Combined: {(accuracy + partial):.2%}")

# Detailed per-category
report = classification_report(
    df_complete['human_label'],
    df_complete['machine_prediction']
)
print(report)
