import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

# -----------------------------
# Load the prepared dataset
# -----------------------------
X = np.load("X.npy")
Y = np.load("Y.npy")

print("Input Shape :", X.shape)
print("Output Shape:", Y.shape)

# -----------------------------
# Train-Test Split (80-20)
# -----------------------------
split = int(len(X) * 0.8)

X_train = X[:split]
X_test = X[split:]

Y_train = Y[:split]
Y_test = Y[split:]

print("Training Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

# -----------------------------
# Build BiLSTM Model
# -----------------------------
model = Sequential()

model.add(Bidirectional(
    LSTM(64, return_sequences=True),
    input_shape=(X_train.shape[1], X_train.shape[2])
))

model.add(Dropout(0.2))

model.add(Bidirectional(
    LSTM(32)
))

model.add(Dropout(0.2))

model.add(Dense(16, activation='relu'))

# Two outputs:
# Solar and Wind
model.add(Dense(2))

# -----------------------------
# Compile
# -----------------------------
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

model.summary()

# -----------------------------
# Callbacks
# -----------------------------
earlystop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    filepath="bilstm_model.h5",
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=False
)

# -----------------------------
# Train
# -----------------------------
history = model.fit(
    X_train,
    Y_train,
    validation_data=(X_test, Y_test),
    epochs=50,
    batch_size=32,
    callbacks=[earlystop, checkpoint],
    verbose=1
)

# -----------------------------
# Save Model
# -----------------------------
model.save("bilstm_model.keras")

print("\nModel Saved Successfully!")

# -----------------------------
# Plot Loss
# -----------------------------
plt.figure(figsize=(10,5))

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("BiLSTM Training")
plt.legend()
plt.grid(True)

plt.savefig("training_loss.png")
plt.show()