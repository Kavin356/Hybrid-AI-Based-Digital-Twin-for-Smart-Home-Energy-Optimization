import numpy as np
import pandas as pd
import joblib

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==========================================
# Load Data
# ==========================================

data = pd.read_csv("normalized_data.csv")

dataset = data[['Solar','Wind']].values

LOOK_BACK = 24

X = []
Y = []

for i in range(len(dataset)-LOOK_BACK):
    X.append(dataset[i:i+LOOK_BACK])
    Y.append(dataset[i+LOOK_BACK])

X = np.array(X)
Y = np.array(Y)

split = int(0.8*len(X))

X_train = X[:split]
Y_train = Y[:split]

X_test = X[split:]
Y_test = Y[split:]

scaler = joblib.load("scaler.save")

results = []

# ==========================================
# Different Architectures
# ==========================================

architectures = {

    "1 Layer":[128],

    "2 Layers":[128,64],

    "3 Layers":[128,64,32]

}

for name,layers in architectures.items():

    print("\nTraining :",name)

    model = Sequential()

    if len(layers)==1:

        model.add(
            Bidirectional(
                LSTM(
                    layers[0],
                    input_shape=(LOOK_BACK,2)
                )
            )
        )

    else:

        model.add(
            Bidirectional(
                LSTM(
                    layers[0],
                    return_sequences=True,
                    input_shape=(LOOK_BACK,2)
                )
            )
        )

        for units in layers[1:-1]:

            model.add(
                Bidirectional(
                    LSTM(
                        units,
                        return_sequences=True
                    )
                )
            )

        model.add(
            Bidirectional(
                LSTM(
                    layers[-1]
                )
            )
        )

    model.add(Dropout(0.2))

    model.add(Dense(32,activation='relu'))

    model.add(Dense(2))

    model.compile(
        optimizer='adam',
        loss='mse'
    )

    early = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    model.fit(

        X_train,
        Y_train,

        validation_data=(X_test,Y_test),

        epochs=30,

        batch_size=32,

        verbose=0,

        callbacks=[early]

    )

    pred = model.predict(X_test,verbose=0)

    actual = scaler.inverse_transform(Y_test)
    pred = scaler.inverse_transform(pred)

    mae = mean_absolute_error(actual,pred)

    rmse = np.sqrt(mean_squared_error(actual,pred))

    r2 = r2_score(actual,pred)

    mape = np.mean(
        np.abs((actual-pred)/(actual+1e-8))
    )*100

    results.append([

        name,

        mae,

        rmse,

        mape,

        r2

    ])

# ==========================================
# Save Performance Matrix
# ==========================================

results = pd.DataFrame(

    results,

    columns=[

        "Architecture",

        "MAE",

        "RMSE",

        "MAPE",

        "R2 Score"

    ]

)

results.to_csv(

    "bilstm_layer_performance.csv",

    index=False

)

print(results)