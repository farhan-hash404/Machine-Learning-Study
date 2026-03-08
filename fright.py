# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Step 1: Create or load dataset
data = {
    "Freight_Price": [100, 150, 200, 250, 300, 350, 400],
    "Distance": [50, 80, 100, 120, 150, 170, 200],
    "Goods_Quantity": [20, 30, 40, 50, 60, 65, 75]
}

df = pd.DataFrame(data)

# Step 2: Select input features
X = df[["Freight_Price", "Distance"]]

# Step 3: Target variable
y = df["Goods_Quantity"]

# Step 4: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 5: Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Predict quantity
predictions = model.predict(X_test)

# Step 7: Check error
error = mean_absolute_error(y_test, predictions)
print("Mean Absolute Error:", error)

# Step 8: Predict new goods quantity
new_data = pd.DataFrame({
    "Freight_Price": [320],
    "Distance": [160]
})

predicted_quantity = model.predict(new_data)

print("Predicted Goods Quantity:", predicted_quantity[0])