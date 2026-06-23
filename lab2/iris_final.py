import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score

# ========================================================
# 1. LOAD MULTICLASS DATASET & PREPROCESS
# ========================================================
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42
)

# Scale features (Critical step for SVM performance)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ========================================================
# 2. BASELINE SVM MODEL
# ========================================================
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train_scaled, y_train)
baseline_acc = accuracy_score(y_test, svm_model.predict(X_test_scaled))

# ========================================================
# 3. MULTICLASS DATA POISONING ATTACK (Label Flipping)
# ========================================================
def poison_multiclass_labels(y, poison_rate=0.2):
    y_poisoned = y.copy()
    n_samples = len(y)
    n_poison = int(poison_rate * n_samples)
    
    np.random.seed(42)
    indices = np.random.choice(n_samples, n_poison, replace=False)
    
    # Multiclass shift: Class 0->1, 1->2, 2->0
    y_poisoned[indices] = (y_poisoned[indices] + 1) % 3
    return y_poisoned

y_train_poisoned = poison_multiclass_labels(y_train, poison_rate=0.2)

# Train on Poisoned Data
svm_model.fit(X_train_scaled, y_train_poisoned)
poisoned_acc = accuracy_score(y_test, svm_model.predict(X_test_scaled))

# ========================================================
# 4. DEFENSE MECHANISM (Isolation Forest)
# ========================================================
iso = IsolationForest(contamination=0.2, random_state=42)
yhat = iso.fit_predict(X_train_scaled)

mask = yhat != -1
X_clean = X_train_scaled[mask]
y_clean = y_train_poisoned[mask]

# Retrain on Cleaned Data
svm_model.fit(X_clean, y_clean)
defense_acc = accuracy_score(y_test, svm_model.predict(X_test_scaled))

# ========================================================
# OUTPUT RESULTS FOR YOUR REPORT TABLE
# ========================================================
print("======= FINAL IRIS MULTICLASS SVM RESULTS =======")
print(f"Baseline Accuracy:        {baseline_acc * 100:.2f}%")
print(f"Poisoned Accuracy:        {poisoned_acc * 100:.2f}%")
print(f"Accuracy After Defense:   {defense_acc * 100:.2f}%")