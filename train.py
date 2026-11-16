
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# 1. Load the dataset
df = pd.read_csv("New_HR-Employee-Attrition.csv")

# 2. Remove unnecessary columns
columns_to_drop = [
    "EmployeeNumber",
    "EmployeeCount",
    "Over18",
    "StandardHours"
]
df = df.drop(columns=columns_to_drop)

# 3. Prepare the target and features
y = df["Attrition"].map({"No": 0, "Yes": 1})
X = df.drop(columns=["Attrition"])

# 4. Identify numerical and categorical features
numerical_columns = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_columns = X.select_dtypes(include=["object", "category"]).columns.tolist()

# 5. Create preprocessing pipelines
numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, numerical_columns),
    ("cat", categorical_pipeline, categorical_columns)
])

# 6. Split the dataset
X_full_train, X_test, y_full_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=1,
    stratify=y
)

# 7. Create the final model
final_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=1
    ))
])

# 8. Train the model
final_model.fit(X_full_train, y_full_train)

# 9. Evaluate on the test dataset
predictions = final_model.predict(X_test)
probabilities = final_model.predict_proba(X_test)[:, 1]

print("FINAL MODEL TEST RESULTS")
print("------------------------")
print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
print("Precision:", round(precision_score(y_test, predictions), 4))
print("Recall:", round(recall_score(y_test, predictions), 4))
print("F1-score:", round(f1_score(y_test, predictions), 4))
print("ROC-AUC:", round(roc_auc_score(y_test, probabilities), 4))

# 10. Save the model separately until results are verified
with open("employee_attrition_model_from_script.pkl", "wb") as file:
    pickle.dump(final_model, file)

print("\nModel trained and saved successfully!")
