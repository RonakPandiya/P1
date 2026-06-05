# ── pip install pandas matplotlib scikit-learn ──────────────────────────
 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
 
# ── Dataset ──────────────────────────────────────────────────────────────
data = {
    "Study_Hours":      [1,  2,  3,  4,  5,  6,  7,  8,  9, 10,  3,  6],
    "Attendance":       [45, 50, 52, 58, 63, 70, 75, 80, 88, 95, 60, 68],
    "Assignment_Score": [30, 35, 40, 50, 55, 65, 70, 78, 85, 92, 45, 62],
    "Performance": ["Low","Low","Low","Medium","Medium","Medium",
                   "High","High","High","High","Low","Medium"]
}
df = pd.DataFrame(data)
print(df.to_string(index=False))
 
# ── Model and Tuning ─────────────────────────────────────────────────────
X = df[["Study_Hours", "Attendance", "Assignment_Score"]]
y = df["Performance"]
 
param_grid = {
    "n_estimators": [20, 50, 100],
    "max_depth":    [2, 3, None],
    "max_features": ["sqrt", None]
}
 
grid = GridSearchCV(RandomForestClassifier(random_state=42),
                    param_grid, cv=3, scoring='accuracy')
grid.fit(X, y)
 
best = grid.best_estimator_
pred = best.predict(X)
acc  = accuracy_score(y, pred)
 
print(f'\nBest Parameters : {grid.best_params_}')
print(f'Accuracy        : {acc:.2f}')
print(f'Predictions     : {list(pred)}')
 
# ── Graph ────────────────────────────────────────────────────────────────
labels = ['Accuracy', 'Error Rate']
values = [acc, 1 - acc]
colors = ['steelblue', 'tomato']
plt.bar(labels, values, color=colors, width=0.4)
plt.ylim(0, 1.2)
for i, v in enumerate(values):
    plt.text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=12)
plt.title("Random Forest: Accuracy vs Error Rate")
plt.ylabel('Score')
plt.grid(axis='y')
plt.tight_layout()
plt.show()
