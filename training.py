import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, mean_absolute_error
from xgboost import XGBClassifier, XGBRegressor

# Load dataset
df = pd.read_csv("injury_data.csv")

# Features
X = df[["Player_Age", "Player_Weight", "Player_Height", "Previous_Injuries", "Training_Intensity"]]

# Targets
y_injury = df["Likelihood_of_Injury"]       # 0 = no injury, 1 = injury
y_recovery = df["Recovery_Time"]   # days/weeks

# Train/test split
X_train, X_test, y_injury_train, y_injury_test, y_recovery_train, y_recovery_test = train_test_split(
    X, y_injury, y_recovery, test_size=0.2, random_state=42
)

# Scale continuous features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Injury Classification Model ---
clf = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
clf.fit(X_train_scaled, y_injury_train)

y_pred_injury = clf.predict(X_test_scaled)
print("Injury F1-score:", f1_score(y_injury_test, y_pred_injury))

# --- Recovery Time Regression Model ---
reg = XGBRegressor(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
reg.fit(X_train_scaled, y_recovery_train)

y_pred_recovery = reg.predict(X_test_scaled)
print("Recovery MAE:", mean_absolute_error(y_recovery_test, y_pred_recovery))


#saving the model
 #import joblib

 # Save classifier, regressor, and scaler
 #joblib.dump(clf, "injury_classifier.joblib")
 #joblib.dump(reg, "recovery_regressor.joblib")
#joblib.dump(scaler, "scaler.joblib")
