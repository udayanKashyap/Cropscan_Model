import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

# Load the CSV file
data_file = "./dataset/dataset.csv"  # Replace with your file path
df = pd.read_csv(data_file)

# Extract features (R, G, B, UV) and target (class)
X = df[["R", "G", "B", "UV"]]
y = df["class"]

# Normalize the features using StandardScaler
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Convert the target variable to categorical if it's not numeric
if y.dtype == "object":
    y = pd.factorize(y)[0]

# Convert y to one-hot encoding (for multi-class classification)
y_onehot = to_categorical(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_normalized, y_onehot, test_size=0.2, random_state=42
)

# Define a neural network model optimized for small datasets
model = Sequential(
    [
        Dense(
            32,
            input_dim=4,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(0.01),
        ),  # Reduced neurons
        Dropout(0.4),  # Increased dropout for regularization
        Dense(16, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        Dense(y_onehot.shape[1], activation="softmax"),  # Output layer
    ]
)

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)

# Train the model with smaller epochs and batch size
history = model.fit(
    X_train,
    y_train,
    epochs=200,  # Allow up to 100 epochs (will stop early if overfitting)
    batch_size=4,  # Smaller batch size for better gradient estimates
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1,
)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {accuracy:.2f}")

# Predict classes
y_pred_probs = model.predict(X_test)
y_pred = y_pred_probs.argmax(axis=1)
y_test_actual = y_test.argmax(axis=1)

# Generate classification report
report = classification_report(y_test_actual, y_pred, output_dict=True)
print("\nClassification Report:\n", classification_report(y_test_actual, y_pred))

# Extract precisions for all classes
classes = list(report.keys())[
    :-3
]  # Skip last 3 summary metrics (accuracy, macro avg, weighted avg)
precisions = [report[cls]["precision"] for cls in classes]

# Plot the precision scores
plt.figure(figsize=(10, 6))
plt.bar(classes, precisions, color="skyblue", alpha=0.8)
plt.xlabel("Classes")
plt.ylabel("Precision")
plt.title("Precision for Each Class")
plt.xticks(rotation=45)
plt.ylim(0, 1)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()
