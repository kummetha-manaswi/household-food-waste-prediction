
# 1. Import libraries
import pandas as pd

# 2. Load dataset
df = pd.read_csv("INT234.csv")

# 2. Common columns
id_cols = [
    "Timestamp", "Email Address", "Name",
    "Age Group", "Household Size",
    "Which best describes you?"
]

food_cols = [
    "Food Group", "Reason Purchased", "Portion Size",
    "People Intended", "Storage Method",
    "Consumption Frequency", "Consumed Percentage",
    "Wasted Percentage", "Waste Reason", "Leftover Action"
]

# 3. First food section (old 37 responses)
old = df.loc[:36, id_cols + list(df.columns[6:16])].copy()
old.columns = id_cols + food_cols

# Remaining responses in first section
new = df.loc[37:, id_cols + [
    df.columns[16], *df.columns[7:16]
]].copy()
new.columns = id_cols + food_cols

food1 = pd.concat([old, new], ignore_index=True)

# 4. Second food section
food2 = df[id_cols + list(df.columns[22:32])].copy()
food2.columns = id_cols + food_cols

# 5. Third food section
food3 = df[id_cols + list(df.columns[33:43])].copy()
food3.columns = id_cols + food_cols

# 6. Combine all sections
combined = pd.concat([food1, food2, food3], ignore_index=True)

# 7. Remove empty records and duplicates
combined = combined.dropna(subset=["Food Group"])
combined = combined.drop_duplicates()

# 8. Final checking
print("Final Shape:", combined.shape)
print("Missing Values:\n", combined.isnull().sum())
print("Duplicates:", combined.duplicated().sum())

print(combined.head(10))
print(df.columns.tolist())
df = combined.copy()

import matplotlib.pyplot as plt
import seaborn as sns
df["Wasted Percentage"].value_counts().plot(kind="bar")
plt.title("Food Waste Distribution")
plt.xlabel("Wasted Percentage")
plt.ylabel("Number of Responses")
plt.show()


sns.countplot(data=df, x="Reason Purchased", hue="Wasted Percentage")
plt.title("Reason Purchased vs Food Waste")
plt.xticks(rotation=45)
plt.show()

# ==========================================
# 4. FOOD GROUP VS FOOD WASTE
# ==========================================
sns.countplot(data=df, x="Food Group", hue="Wasted Percentage")
plt.title("Food Group vs Food Waste")
plt.xticks(rotation=45)
plt.show()

# ==========================================
# 5. STORAGE METHOD VS FOOD WASTE
# ==========================================
sns.countplot(data=df, x="Storage Method", hue="Wasted Percentage")
plt.title("Storage Method vs Food Waste")
plt.xticks(rotation=45)
plt.show()

# ==========================================
# 6. REASONS FOR FOOD WASTE
# ==========================================
df["Waste Reason"].value_counts().plot(kind="bar")
plt.title("Reasons for Food Waste")
plt.xticks(rotation=45)
plt.show()

print(df["Wasted Percentage"].value_counts())

# ==========================================
# 7. SELECT INPUT AND OUTPUT
# ==========================================
X = df.drop(columns=[
    "Wasted Percentage",
    "Timestamp",
    "Email Address",
    "Name",
    "Consumed Percentage",
    "Waste Reason",
    "Leftover Action"
])

y = df["Wasted Percentage"]

# Convert categorical data into numbers
X = pd.get_dummies(X, dtype=int)

print("Input features:", X.shape)
print("Target:", y.shape)

# ==========================================
# 8. SPLIT TRAINING AND TESTING DATA
# ==========================================
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

# ==========================================
# 9. DECISION TREE
# ==========================================
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, zero_division=0))

# ==========================================
# 10. RANDOM FOREST
# ==========================================
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))
print(classification_report(y_test, rf_pred, zero_division=0))

# ==========================================
# 11. LOGISTIC REGRESSION
# ==========================================
from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)

print("Logistic Regression Accuracy:", accuracy_score(y_test, lr_pred))
print(classification_report(y_test, lr_pred, zero_division=0))

# ==========================================
# 12. SUPPORT VECTOR MACHINE
# ==========================================
from sklearn.svm import SVC

svm_model = SVC()
svm_model.fit(X_train, y_train)

svm_pred = svm_model.predict(X_test)

print("SVM Accuracy:", accuracy_score(y_test, svm_pred))
print(classification_report(y_test, svm_pred, zero_division=0))

# ==========================================
# 13. K NEAREST NEIGHBORS
# ==========================================
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

knn_model = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=5)
)

knn_model.fit(X_train, y_train)
knn_pred = knn_model.predict(X_test)

print("KNN Accuracy:", round(accuracy_score(y_test, knn_pred) * 100, 2), "%")
print(classification_report(y_test, knn_pred, zero_division=0))

# ==========================================
# 14. ACCURACY COMPARISON GRAPH
# ==========================================
models = ["Decision Tree", "Random Forest", "Logistic Regression", "SVM", "KNN"]
accuracy = [
    accuracy_score(y_test, y_pred) * 100,
    accuracy_score(y_test, rf_pred) * 100,
    accuracy_score(y_test, lr_pred) * 100,
    accuracy_score(y_test, svm_pred) * 100,
    accuracy_score(y_test, knn_pred) * 100
]

plt.figure(figsize=(9, 5))
bars = plt.bar(models, accuracy)

plt.title("Accuracy Comparison of Machine Learning Models")
plt.xlabel("Machine Learning Models")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)
plt.xticks(rotation=25)

for bar in bars:
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 1,
             f"{bar.get_height():.2f}%",
             ha="center")

plt.tight_layout()
plt.show()

# ==========================================
# 15. F1 SCORE COMPARISON
# ==========================================
from sklearn.metrics import f1_score

f1 = [
    f1_score(y_test, y_pred, average="macro", zero_division=0),
    f1_score(y_test, rf_pred, average="macro", zero_division=0),
    f1_score(y_test, lr_pred, average="macro", zero_division=0),
    f1_score(y_test, svm_pred, average="macro", zero_division=0),
    f1_score(y_test, knn_pred, average="macro", zero_division=0)
]

plt.figure(figsize=(9, 5))
bars = plt.bar(models, f1)

plt.title("Macro F1-Score Comparison of Machine Learning Models")
plt.xlabel("Machine Learning Models")
plt.ylabel("Macro F1-Score")
plt.xticks(rotation=25)

for bar in bars:
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.01,
             f"{bar.get_height():.2f}",
             ha="center")

plt.tight_layout()
plt.show()

# ==========================================
# 16. MODEL PERFORMANCE TABLE
# ==========================================
from sklearn.metrics import precision_score, recall_score

performance_df = pd.DataFrame({
    "Model": models,
    "Accuracy (%)": accuracy,
    "Macro Precision": [
        precision_score(y_test, p, average="macro", zero_division=0)
        for p in [y_pred, rf_pred, lr_pred, svm_pred, knn_pred]
    ],
    "Macro Recall": [
        recall_score(y_test, p, average="macro", zero_division=0)
        for p in [y_pred, rf_pred, lr_pred, svm_pred, knn_pred]
    ],
    "Macro F1-Score": f1
})

print(performance_df)

