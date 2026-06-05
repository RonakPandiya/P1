# ── pip install pandas matplotlib scikit-learn ──────────────────────────
 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
 
# ── Dataset ──────────────────────────────────────────────────────────────
data = {
    "Marks":      [32, 36, 40, 45, 55, 60, 68, 72, 78, 85],
    "Attendance": [45, 48, 52, 58, 62, 68, 73, 78, 84, 90]
}
df = pd.DataFrame(data)
print(df.to_string(index=False))
 
# ── Scale data ───────────────────────────────────────────────────────────
scaler = StandardScaler()
X = scaler.fit_transform(df)
 
# ── Silhouette scores for different k ────────────────────────────────────
print('\nSilhouette Scores:')
best_k, best_score = 2, -1
for k in [2, 3, 4]:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    score  = silhouette_score(X, labels)
    print(f'  k = {k}  -->  {score:.3f}')
    if score > best_score:
        best_k, best_score = k, score
 
print(f'\nBest k = {best_k}  (Silhouette = {best_score:.3f})')
 
# ── Final model with best k ───────────────────────────────────────────────
final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df["Cluster"] = final.fit_predict(X)
print(df.to_string(index=False))
 
# ── Graph ────────────────────────────────────────────────────────────────
colors = ['steelblue', 'tomato', 'green', 'purple']
for c in df['Cluster'].unique():
    sub = df[df['Cluster'] == c]
    plt.scatter(sub['Marks'], sub['Attendance'],
                label=f'Cluster {c}', s=80, color=colors[c])
plt.xlabel("Marks")
plt.ylabel("Attendance")
plt.title(f"K-Means Clustering (k={best_k}): Student Groups")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
