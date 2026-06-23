import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ==========================================
# PART A: BASELINE MODEL
# ==========================================
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.3, random_state=42
)

model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Baseline Accuracy:        ", accuracy_score(y_test, y_pred))


# ==========================================
# PART B: DATA POISONING ATTACK (Label Flipping)
# ==========================================
def poison_labels(y, poison_rate=0.2):  # Set to 0.2 (20%) as per manual
    y_poisoned = y.copy()
    n_samples = len(y)
    n_poison = int(poison_rate * n_samples)
    
    # Randomly pick indices to poison without replacement
    np.random.seed(42)  # Set seed for reproducible results
    indices = np.random.choice(n_samples, n_poison, replace=False)
    
    # Flip the binary labels: (1 - 0 = 1) and (1 - 1 = 0)
    y_poisoned[indices] = 1 - y_poisoned[indices]
    
    return y_poisoned

# Apply 20% poisoning to training data
y_train_poisoned = poison_labels(y_train, poison_rate=0.2)

# Retrain the model using the POISONED training data
poisoned_model = LogisticRegression(max_iter=5000)
poisoned_model.fit(X_train, y_train_poisoned)

# Evaluate performance degradation on the clean test data
y_pred_poisoned = poisoned_model.predict(X_test)
print("Poisoned Model Accuracy: ", accuracy_score(y_test, y_pred_poisoned))