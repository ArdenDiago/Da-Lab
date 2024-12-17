import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the data
titanic_data = pd.read_csv("./data.csv")

# Transforming the data
titanic_data = titanic_data.dropna(subset=["Age", "Fare"])

def draw_graph(x, y, color, alpha=0.5, labels=[], plot=None):
    plt.scatter(x, y, color=color, alpha=alpha)
    if plot:
        X_val, y_val, line_color, label = plot
        plt.plot(X_val, y_val, color=line_color, label=label)
    plt.title(labels[0])
    plt.xlabel(labels[1])
    plt.ylabel(labels[2])
    if plot:
        plt.legend()
    plt.show()

# Scatter Plot
age, fare = titanic_data["Age"], titanic_data["Fare"]
draw_graph(age, fare, color="blue", labels=["Scatter Plot: Age vs Fare", "Age", "Fare"])

# Pearson Correlation Coefficient
corr_coeff = age.corr(fare)
print(f"\nCorrelation Coefficient between Age & Fare: {corr_coeff}")

# Linear Regression
X = titanic_data[["Age"]]  # Independent variable needs to be 2D for sklearn
y = titanic_data["Fare"]

linear_model = LinearRegression()
linear_model.fit(X, y)

y_pred = linear_model.predict(X)

r_squared = linear_model.score(X, y)
print(f"\nR-Squared: {r_squared}")

new_age = pd.DataFrame({"Age": [25, 30, 40]})
pred_fares = linear_model.predict(new_age)
print(f"\nPredicted fares for ages 25, 30, 40: {pred_fares}")

draw_graph(X, y, color="blue",labels=["Linear Regression: Age vs Fare", "Age", "Fare"], plot=(X, y_pred, "red", "Regression Line"))

# Logistic Regression
titanic_data = titanic_data.dropna(subset=["Survived"])
X = titanic_data[["Age"]]
y = titanic_data["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

logreg_model = LogisticRegression()
logreg_model.fit(X_train, y_train)

y_pred = logreg_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy of the model: {accuracy}")

predicted_survival = logreg_model.predict(new_age)
print(f"\nPredicted Survival for ages 25, 30, 40: {predicted_survival}")

X_vals = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_vals = logreg_model.predict_proba(X_vals)[:, 1]

draw_graph(X["Age"],y,color="blue",labels=["Logistic Regression: Age vs Survival Probability","Age","Survival Probability",],plot=(X_vals, y_vals, "red", "Logistic Regression Curve"))