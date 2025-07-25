# Train and save model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib
import mlflow
import mlflow.sklearn
from data_loader import fetch_data 

df = fetch_data()
if df.empty:
    raise Exception("No data found in MongoDB!")
X = df.drop(columns=["monthly_charge", "customer_id"]) #i/p
y = df["monthly_charge"] #o/p
#print(X)
#print(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train) # Building model / Training model

score = model.score(X_train, y_train)

# Save model
joblib .dump(model, "model.pkl")

# Log to MLflow
with mlflow.start_run():
    mlflow.sklearn.log_model(model, "linear_model")
    mlflow.log_metric("r2_score", score)

print(f"Linear Regression model saved. R2 Score: {score:.2f}")