# Required libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Input, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# =====================
# Load and Preprocess Data
# =====================
# Load datasets
batting_df = pd.read_csv("batting.csv")
bowling_df = pd.read_csv("bowling.csv")

# Clean up bowling data
bowling_df['match'] = bowling_df['match'].ffill().astype(int)
bowling_df['Overs'] = bowling_df['Overs'].astype(float)

# Merge on Player, Team, match
combined_df = pd.merge(
    batting_df,
    bowling_df,
    on=['Player', 'Team', 'match'],
    how='outer',
    suffixes=('_bat', '_bowl')
)

# Fill NaNs for numerical columns
numeric_cols = combined_df.select_dtypes(include='number').columns
combined_df[numeric_cols] = combined_df[numeric_cols].fillna(0)

# =====================
# Feature Engineering
# =====================
combined_df['BattingImpact'] = (
    (combined_df['Runs_bat'] * combined_df['SR']) / 100 +
    (combined_df['4s'] * 4) +
    (combined_df['6s'] * 6)
)

combined_df['BoundaryRate'] = (
    (combined_df['4s'] * 0.6 + combined_df['6s'] * 0.4) / (combined_df['Balls'] + 1e-6)
)

combined_df['DotBallRate'] = combined_df['Dots'] / (combined_df['Overs'] * 6 + 1e-6)

combined_df['BowlingImpact'] = (
    (combined_df['Wickets'] * 25) / (combined_df['Economy'] + 1)
)

combined_df['PitchScore'] = (
    (combined_df['SR'] * 0.4) +
    (combined_df['BoundaryRate'] * 0.3) +
    (combined_df['DotBallRate'] * 0.3)
)

# =====================
# Aggregate Player Stats
# =====================
player_stats = combined_df.groupby(['Player', 'Team']).agg({
    'BattingImpact': 'mean',
    'BoundaryRate': 'mean',
    'BowlingImpact': 'mean',
    'DotBallRate': 'mean',
    'PitchScore': 'mean',
    'Wickets': 'sum',
    'Runs_bat': 'sum',
    'Overs': 'sum',
    'Balls': 'sum',
    'match': 'count'
}).reset_index()

# Add label: PlayXI
player_stats['PlayXI'] = (player_stats['match'] > 5).astype(int)

# =====================
# Prepare Features and Labels
# =====================
features = ['BattingImpact', 'BoundaryRate', 'BowlingImpact', 'DotBallRate', 'PitchScore', 'Wickets', 'Runs_bat', 'Overs', 'Balls']
X = player_stats[features].values
y = player_stats['PlayXI'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Reshape for RNN
X_train_rnn = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test_rnn = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# =====================
# Callbacks
# =====================
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=0.00001
)

dnn_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'best_dnn_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max'
)

rnn_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'best_rnn_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max'
)

# =====================
# DNN Model
# =====================
dnn_model = Sequential([
    Dense(1024, input_shape=(X_train.shape[1],), activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# For both DNN and RNN models, update the metrics parameter in compile()
dnn_model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
)

dnn_model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=16,
    validation_split=0.2,
    callbacks=[early_stopping, reduce_lr, dnn_checkpoint],
    verbose=1
)

# =====================
# RNN Model (LSTM)
# =====================
rnn_model = Sequential([
    LSTM(512, input_shape=(1, X_train.shape[1]), return_sequences=True),
    BatchNormalization(),
    Dropout(0.4),
    LSTM(256, return_sequences=True),
    BatchNormalization(),
    Dropout(0.3),
    LSTM(128, return_sequences=True),
    BatchNormalization(),
    Dropout(0.3),
    LSTM(64),
    BatchNormalization(),
    Dropout(0.2),
    Dense(32, activation='relu'),
    BatchNormalization(),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

rnn_model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
)

rnn_model.fit(
    X_train_rnn, y_train,
    epochs=150,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping, reduce_lr, rnn_checkpoint],
    verbose=1
)


def select_balanced_team(player_stats, n_batsmen=8, n_bowlers=3):
    # Enhanced batting criteria for Jaipur pitch (known for batting friendly conditions)
    player_stats['BattingScore'] = (
        player_stats['BattingImpact'] * 0.6 +     # Increased batting impact weight
        player_stats['BoundaryRate'] * 40 +       # Higher weight for boundary hitting
        player_stats['Runs_bat'] * 0.3 +          # Consider total runs
        player_stats['PitchScore'] * 20           # Factor in pitch adaptation
    )
    
    # Enhanced bowling criteria for Jaipur conditions
    player_stats['BowlingScore'] = (
        player_stats['BowlingImpact'] * 0.5 +
        (1 - player_stats['Economy']) * 30 +      # Lower economy is better
        player_stats['DotBallRate'] * 20          # Important for control
    )
    
    # Select specialists
    batsmen = player_stats[player_stats['BowlingImpact'] < 15].nlargest(n_batsmen, 'BattingScore')
    bowlers = player_stats[player_stats['BattingScore'] < 25].nlargest(n_bowlers, 'BowlingScore')
    
    return pd.concat([batsmen, bowlers])

# Make predictions
player_stats['PredictionScore'] = dnn_model.predict(X_scaled)
jaipur_xi = select_balanced_team(player_stats)

# Display results with detailed stats
print("\n🏏 Predicted Best XI for Jaipur Match:")
print("\nBatsmen:")
print(jaipur_xi[jaipur_xi['BowlingImpact'] < 15][
    ['Player', 'Team', 'BattingScore', 'BoundaryRate', 'PitchScore']
].round(2))

print("\nBowlers:")
print(jaipur_xi[jaipur_xi['BattingScore'] < 25][
    ['Player', 'Team', 'BowlingScore', 'Economy', 'DotBallRate']
].round(2))

# Save predictions
jaipur_xi.to_csv('e:\\clg\\New folder\\jaipur_match_predictions.csv', index=False)
