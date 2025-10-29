#type: ignore

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 1. Generate fake dataset
np.random.seed(42)
days = pd.date_range(start='2024-01-01', periods=180)
temperature = np.random.normal(loc=20, scale=10, size=180)  # Avg 20°C
weekday = [day.weekday() for day in days]
previous_kwh = np.random.normal(loc=30, scale=5, size=180)

# Fake energy consumption influenced by temperature, weekday, and previous usage
energy_kwh = (
    0.5 * temperature +
    -0.3 * np.array(weekday) +
    0.7 * previous_kwh +
    np.random.normal(loc=0, scale=3, size=180)
)

# Build the DataFrame
df = pd.DataFrame({
    'temperature': temperature,
    'weekday': weekday,
    'previous_kwh': previous_kwh,
    'energy_kwh': energy_kwh
})

# 2. Train-test split
X = df[['temperature', 'weekday', 'previous_kwh']]
y = df['energy_kwh']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# 5. Plot actual vs predicted
plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label="Actual", marker='o')
plt.plot(y_pred, label="Predicted", marker='x')
plt.title("Energy Consumption: Actual vs. Predicted")
plt.xlabel("Sample")
plt.ylabel("kWh")
plt.legend()
plt.tight_layout()
plt.show()