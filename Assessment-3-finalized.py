import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import PowerTransformer

# Load the dataset
data = pd.read_csv('po2_data.csv')

# Show data shape
print("Data Shape:")
print(data.shape)  # This will display the number of rows and columns

# Data info
print("\nData Info:")
print(data.info())  # This will display data types and non-null counts

# Check for missing values
print("\nMissing Values:")
missing_values = data.isnull().sum()
print(missing_values)

# Describe the data
print("\nData Description:")
print(data.describe())

# Pairplot for selected features
selected_features = [
    'age', 'sex', 'jitter(%)', 'jitter(abs)', 'jitter(rap)', 'jitter(ppq5)', 'jitter(ddp)',
    'shimmer(%)', 'shimmer(abs)', 'shimmer(apq3)', 'shimmer(apq5)', 'shimmer(apq11)', 'shimmer(dda)',
    'nhr', 'hnr', 'rpde', 'dfa', 'ppe'
]

# Calculate correlation with target variables
correlation_motor = data[selected_features].corrwith(data['motor_updrs'])
correlation_total = data[selected_features].corrwith(data['total_updrs'])

# Set a correlation threshold for feature selection
correlation_threshold = 0.1  # Features within this threshold will be selected 

# Select features with correlation above the threshold for both targets
selected_features = [
    feature for feature in selected_features
    if abs(correlation_motor[feature]) > correlation_threshold
    and abs(correlation_total[feature]) > correlation_threshold
]
print(f"The featues with relatively higher corelation to target variables are: {selected_features}\n")

# Feature Engineering: Calculate rate of change in UPDRS scores over time
# Because a particular subject is tested multiple times within a set time-period

data['motor_updrs_rate'] = data.groupby('subject#')['motor_updrs'].diff()
data['total_updrs_rate'] = data.groupby('subject#')['total_updrs'].diff()

# Remove NaN values resulting from the rate calculations
data.dropna(subset=['motor_updrs_rate', 'total_updrs_rate'], inplace=True)

# Update the selected features to include the rate features
selected_features += ['motor_updrs_rate', 'total_updrs_rate']

# Update X, y_motor, and y_total with the selected features
X = data[selected_features]
y_motor = data['motor_updrs']
y_total = data['total_updrs']

# Split the data into training and testing sets (60% training and 40% test)
X_train, X_test, y_motor_train, y_motor_test = train_test_split(X, y_motor, test_size=0.4, random_state=1)
X_train, X_test, y_total_train, y_total_test = train_test_split(X, y_total, test_size=0.4, random_state=1)

# Build regression models
motor_model = LinearRegression()
motor_model.fit(X_train, y_motor_train)
total_model = LinearRegression()
total_model.fit(X_train, y_total_train)

# Make predictions
motor_predictions = motor_model.predict(X_test)
total_predictions = total_model.predict(X_test)

# Calculate evaluation metrics for the model
def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    nrmse = rmse / (np.max(y_true) - np.min(y_true))
    r2 = r2_score(y_true, y_pred)
    adjusted_r2 = 1 - (1 - r2) * (len(y_true) - 1) / (len(y_true) - len(selected_features) - 1)
    return mae, mse, rmse, nrmse, r2, adjusted_r2

motor_mae, motor_mse, motor_rmse, motor_nrmse, motor_r2, motor_adjusted_r2 = evaluate_model(y_motor_test, motor_predictions)
total_mae, total_mse, total_rmse, total_nrmse, total_r2, total_adjusted_r2 = evaluate_model(y_total_test, total_predictions)

# Report results
print("Motor UPDRS Prediction Metrics with 60% training and 40% test:")
print(f"Mean Absolute Error: {motor_mae}")
print(f"Mean Squared Error: {motor_mse}")
print(f"Root Mean Squared Error (RMSE): {motor_rmse}")
print(f" Normalized Root Mean Squared Error (NRMSE): {motor_nrmse}")
print(f"R-squared: {motor_r2}")
print(f"Adjusted R-squared: {motor_adjusted_r2}")

print("\nTotal UPDRS Prediction Metrics:")
print(f"Mean Absolute Error: {total_mae}")
print(f"Mean Squared Error: {total_mse}")
print(f"Root Mean Squared Error (RMSE): {total_rmse}")
print(f" Normalized Root Mean Squared Error ( Normalized Root Mean Squared Error (NRMSE)): {total_nrmse}")
print(f"R-squared: {total_r2}")
print(f"Adjusted R-squared: {total_adjusted_r2}\n")

## Task 2 ##

# Define a list of different train-test split ratios
split_ratios = [0.5, 0.6, 0.7, 0.8]

# Initialize lists to store the evaluation results
motor_metrics = []
total_metrics = []

# Initialize variables to track the best split scenario
best_motor_r2 = -1
best_total_r2 = -1
best_split_ratio = None

# Metric names for labels
metric_names = ["Mean Absolute Error", "MSE", "RMSE", "NRMSE", "R-squared", "Adjusted R-squared"]

# Iterate through the split ratios and evaluate models
for split_ratio in split_ratios:
    # Split the data into training and testing sets based on the current ratio
    X_train, X_test, y_motor_train, y_motor_test = train_test_split(X, y_motor, test_size=(1 - split_ratio), random_state=1)
    X_train, X_test, y_total_train, y_total_test = train_test_split(X, y_total, test_size=(1 - split_ratio), random_state=1)

    # Initialize and train linear regression models
    motor_model = LinearRegression()
    motor_model.fit(X_train, y_motor_train)
    total_model = LinearRegression()
    total_model.fit(X_train, y_total_train)

    # Make predictions
    motor_predictions = motor_model.predict(X_test)
    total_predictions = total_model.predict(X_test)

    # Calculate evaluation metrics for motor UPDRS
    motor_mae = mean_absolute_error(y_motor_test, motor_predictions)
    motor_mse = mean_squared_error(y_motor_test, motor_predictions)
    motor_rmse = np.sqrt(motor_mse)
    motor_nrmse = motor_rmse / (np.max(y_motor_test) - np.min(y_motor_test))
    motor_r2 = r2_score(y_motor_test, motor_predictions)
    motor_adjusted_r2 = 1 - (1 - motor_r2) * (len(y_motor_test) - 1) / (len(y_motor_test) - len(selected_features) - 1)

    # Calculate evaluation metrics for total UPDRS
    total_mae = mean_absolute_error(y_total_test, total_predictions)
    total_mse = mean_squared_error(y_total_test, total_predictions)
    total_rmse = np.sqrt(total_mse)
    total_nrmse = total_rmse / (np.max(y_total_test) - np.min(y_total_test))
    total_r2 = r2_score(y_total_test, total_predictions)
    total_adjusted_r2 = 1 - (1 - total_r2) * (len(y_total_test) - 1) / (len(y_total_test) - len(selected_features) - 1)

    # Store the metrics for the current split ratio
    motor_metrics.append((motor_mae, motor_mse, motor_rmse, motor_nrmse, motor_r2, motor_adjusted_r2))
    total_metrics.append((total_mae, total_mse, total_rmse, total_nrmse, total_r2, total_adjusted_r2))

    # Check if this split scenario has better R-squared values for both motor and total UPDRS
    if motor_r2 > best_motor_r2 and total_r2 > best_total_r2:
        best_motor_r2 = motor_r2
        best_total_r2 = total_r2
        best_split_ratio = split_ratio

# Print the evaluation metrics for different split scenarios
for i, split_ratio in enumerate(split_ratios):
    print(f"Split Ratio: {split_ratio * 100}%")
    print("Motor UPDRS Metrics:")
    for j, metric_name in enumerate(metric_names):
        print(f"{metric_name}: {motor_metrics[i][j]}")
    print("\nTotal UPDRS Metrics:")
    for j, metric_name in enumerate(metric_names):
        print(f"{metric_name}: {total_metrics[i][j]}")
    print("-" * 40)

# Determine the best split scenario for further evaluation
if best_split_ratio is not None:
    print(f"\nThe best split scenario with better accuracy for both Motor and Total UPDRS:")
    print(f"Split Ratio: {best_split_ratio * 100}%")
    print(f"Best Motor R-squared: {best_motor_r2}")
    print(f"Best Total R-squared: {best_total_r2}\n")
else:
    print("No split scenario found with better accuracy for both Motor and Total UPDRS.")
    
    
# Scatter Plots of Actual Values vs. Predicted Values

# Actual vs. Predicted Values for Motor UPDRS
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(y_motor_test, motor_predictions, alpha=0.5, c='blue', label='Actual', marker='o')
plt.scatter(y_motor_test, motor_predictions, alpha=0.5, c='red', label='Predicted', marker='x')
plt.plot([min(y_motor_test), max(y_motor_test)], [min(y_motor_test), max(y_motor_test)], color='black', linestyle='--', linewidth=2, label='Actual = Predicted Line')
plt.title("Actual vs. Predicted Values (Motor UPDRS)")
plt.xlabel("Actual Motor UPDRS")
plt.ylabel("Predicted Motor UPDRS")
plt.legend()

# Actual vs. Predicted Values for Total UPDRS
plt.subplot(1, 2, 2)
plt.scatter(y_total_test, total_predictions, alpha=0.5, c='blue', label='Actual', marker='o')
plt.scatter(y_total_test, total_predictions, alpha=0.5, c='red', label='Predicted', marker='x')
plt.plot([min(y_total_test), max(y_total_test)], [min(y_total_test), max(y_total_test)], color='black', linestyle='--', linewidth=2, label='Actual = Predicted Line')
plt.title("Actual vs. Predicted Values (Total UPDRS)")
plt.xlabel("Actual Total UPDRS")
plt.ylabel("Predicted Total UPDRS")
plt.legend()

plt.tight_layout()
plt.show()

  
# Residual Analysis
motor_residuals = y_motor_test - motor_predictions
total_residuals = y_total_test - total_predictions

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(motor_predictions, motor_residuals, alpha=0.5, c='r', label='Predicted', marker='o')
plt.scatter(motor_predictions, y_motor_test, alpha=0.5, c='b', label='Actual', marker='x')
plt.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Zero Residual Line')
plt.title("Residuals vs. Predicted Values (Motor UPDRS)")
plt.xlabel("Motor UPDRS Predicted Values")
plt.ylabel("Residuals")
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(total_predictions, total_residuals, alpha=0.5, c='r', label='Predicted', marker='o')
plt.scatter(total_predictions, y_total_test, alpha=0.5, c='b', label='Actual', marker='x')
plt.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Zero Residual Line')
plt.title("Residuals vs. Predicted Values (Total UPDRS)")
plt.xlabel("Total UPDRS Predicted Values")
plt.ylabel("Residuals")
plt.legend()

plt.tight_layout()
plt.show()

## Task-3 ##

# Apply log-transform to the target variables (Motor UPDRS and Total UPDRS)
y_motor = np.log1p(y_motor)
y_total = np.log1p(y_total)

# Re-split the data into training and testing sets with the best split ratio
X_train, X_test, y_motor_train, y_motor_test = train_test_split(X, y_motor, test_size=best_split_ratio, random_state=1)
X_train, X_test, y_total_train, y_total_test = train_test_split(X, y_total, test_size=best_split_ratio, random_state=1)

# Rebuild the linear regression models with log-transformed targets
motor_model = LinearRegression()
motor_model.fit(X_train, y_motor_train)

total_model = LinearRegression()
total_model.fit(X_train, y_total_train)

# Make predictions on the test set
motor_predictions = motor_model.predict(X_test)
total_predictions = total_model.predict(X_test)

# Inverse log-transform to get back the original scale for evaluation
motor_predictions = np.expm1(motor_predictions)
total_predictions = np.expm1(total_predictions)

# Calculate evaluation metrics for Motor UPDRS
motor_mae, motor_mse, motor_rmse, motor_nrmse, motor_r2, motor_adjusted_r2 = evaluate_model(np.expm1(y_motor_test), motor_predictions)

# Calculate evaluation metrics for Total UPDRS
total_mae, total_mse, total_rmse, total_nrmse, total_r2, total_adjusted_r2 = evaluate_model(np.expm1(y_total_test), total_predictions)

# Report results with log-transform
print("Motor UPDRS Prediction Metrics with Log-Transform:")
print(f"Mean Absolute Error: {motor_mae}")
print(f"Mean Squared Error: {motor_mse}")
print(f"Root Mean Squared Error (RMSE): {motor_rmse}")
print(f" Normalized Root Mean Squared Error (NRMSE): {motor_nrmse}")
print(f"R-squared: {motor_r2}")
print(f"Adjusted R-squared: {motor_adjusted_r2}")

print("\nTotal UPDRS Prediction Metrics with Log-Transform:")
print(f"Mean Absolute Error: {total_mae}")
print(f"Mean Squared Error: {total_mse}")
print(f"Root Mean Squared Error (RMSE): {total_rmse}")
print(f" Normalized Root Mean Squared Error (NRMSE): {total_nrmse}")
print(f"R-squared: {total_r2}")
print(f"Adjusted R-squared: {total_adjusted_r2}")


# Define evaluation metrics before and after log transformation
metrics_before_log = [motor_mae, motor_mse, motor_rmse, motor_nrmse, motor_r2, motor_adjusted_r2]
metrics_after_log = [total_mae, total_mse, total_rmse, total_nrmse, total_r2, total_adjusted_r2]

# Visualise how the transformed metrices compare to previous ones
# Metric names for labels
metric_names = ["MAE", "MSE", "RMSE", "NRMSE", "R-squared", "Adjusted R-squared"]

# Create a bar chart to compare metrics before and after log transformation
fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.35
index = range(len(metric_names))

bar1 = ax.bar(index, metrics_before_log, bar_width, label='Before Log-Transform', alpha=0.7)
bar2 = ax.bar([i + bar_width for i in index], metrics_after_log, bar_width, label='After Log-Transform', alpha=0.7)

# Set labels, title, and legend
ax.set_xlabel('Metrics')
ax.set_ylabel('Values')
ax.set_title('Comparison of Metrics Before and After Log-Transform')
ax.set_xticks([i + bar_width / 2 for i in index])
ax.set_xticklabels(metric_names)
ax.legend()

# Show the bar chart
plt.tight_layout()
plt.show()

## Task-4 ##

# Initialize the PowerTransformer for Yeo-Johnson transformation
power_transformer = PowerTransformer(method='yeo-johnson')

# Fit and transform the selected features
X_transformed = power_transformer.fit_transform(X)

# Split the transformed data into training and testing sets (60% training and 40% test)
X_train, X_test, y_motor_train, y_motor_test = train_test_split(X_transformed, y_motor, test_size=0.4, random_state=1)
X_train, X_test, y_total_train, y_total_test = train_test_split(X_transformed, y_total, test_size=0.4, random_state=1)

# Rebuild the linear regression models with transformed features
motor_model_transformed = LinearRegression()
motor_model_transformed.fit(X_train, y_motor_train)

total_model_transformed = LinearRegression()
total_model_transformed.fit(X_train, y_total_train)

# Make predictions on the test set with transformed features
motor_predictions_transformed = motor_model_transformed.predict(X_test)
total_predictions_transformed = total_model_transformed.predict(X_test)

# Calculate evaluation metrics for Motor UPDRS with transformed features
motor_mae_transformed, motor_mse_transformed, motor_rmse_transformed, motor_nrmse_transformed, motor_r2_transformed, motor_adjusted_r2_transformed = evaluate_model(y_motor_test, motor_predictions_transformed)

# Calculate evaluation metrics for Total UPDRS with transformed features
total_mae_transformed, total_mse_transformed, total_rmse_transformed, total_nrmse_transformed, total_r2_transformed, total_adjusted_r2_transformed = evaluate_model(y_total_test, total_predictions_transformed)

# Report results with feature transformation using Yeo-Johnson
print("\nMotor UPDRS Prediction Metrics with Yeo-Johnson Transformation:")
print(f"Mean Absolute Error: {motor_mae_transformed}")
print(f"Mean Squared Error: {motor_mse_transformed}")
print(f"Root Mean Squared Error (RMSE): {motor_rmse_transformed}")
print(f" Normalized Root Mean Squared Error (NRMSE): {motor_nrmse_transformed}")
print(f"R-squared: {motor_r2_transformed}")
print(f"Adjusted R-squared: {motor_adjusted_r2_transformed}")

print("\nTotal UPDRS Prediction Metrics with Yeo-Johnson Transformation:")
print(f"Mean Absolute Error: {total_mae_transformed}")
print(f"Mean Squared Error: {total_mse_transformed}")
print(f"Root Mean Squared Error (RMSE): {total_rmse_transformed}")
print(f" Normalized Root Mean Squared Error (NRMSE): {total_nrmse_transformed}")
print(f"R-squared: {total_r2_transformed}")
print(f"Adjusted R-squared: {total_adjusted_r2_transformed}")

# Visualise how the transformed metrices compare to previous ones

# Define metric names for comparison
metric_names = ["MAE", "MSE", "RMSE", "NRMSE", "R-squared", "Adjusted R-squared"]

# Create lists of metrics before and after transformation for Motor UPDRS
motor_metrics_before = [motor_mae, motor_mse, motor_rmse, motor_nrmse, motor_r2, motor_adjusted_r2]
motor_metrics_after = [motor_mae_transformed, motor_mse_transformed, motor_rmse_transformed, motor_nrmse_transformed, motor_r2_transformed, motor_adjusted_r2_transformed]

# Create lists of metrics before and after transformation for Total UPDRS
total_metrics_before = [total_mae, total_mse, total_rmse, total_nrmse, total_r2, total_adjusted_r2]
total_metrics_after = [total_mae_transformed, total_mse_transformed, total_rmse_transformed, total_nrmse_transformed, total_r2_transformed, total_adjusted_r2_transformed]

# Create subplots for Motor UPDRS metrics
plt.figure(figsize=(15, 5))
for i, metric_name in enumerate(metric_names):
    plt.subplot(2, len(metric_names), i+1)
    plt.bar(['Before', 'After'], [motor_metrics_before[i], motor_metrics_after[i]], color=['blue', 'green'])
    plt.title(f"Motor UPDRS {metric_name}")

# Create subplots for Total UPDRS metrics
for i, metric_name in enumerate(metric_names):
    plt.subplot(2, len(metric_names), len(metric_names)+i+1)
    plt.bar(['Before', 'After'], [total_metrics_before[i], total_metrics_after[i]], color=['blue', 'green'])
    plt.title(f"Total UPDRS {metric_name}")

plt.tight_layout()
plt.show()
