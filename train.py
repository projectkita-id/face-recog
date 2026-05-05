import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import pickle
import os

print("="*60)
print("🚀 TRAINING LSTM MODEL - GESTURE RECOGNITION (neutral + A-Z)")
print("="*60)

# ============================================================
# 1. Load semua dataset (neutral + A-Z)
# ============================================================
print("\n[1/6] Loading dataset...")

labels = ['neutral'] + [chr(i) for i in range(ord('A'), ord('D') + 1)]
dfs = {}
missing_files = []

for lbl in labels:
    fname = f"{lbl}.txt"
    if os.path.exists(fname):
        df = pd.read_csv(fname)
        dfs[lbl] = df
        print(f"✅ {lbl:7s}: {len(df)} frames -> {fname}")
    else:
        missing_files.append(fname)

if missing_files:
    print("\n❌ ERROR: Ada file yang belum ada:")
    for f in missing_files:
        print(f"   - {f}")
    print("\n   Pastikan semua gesture sudah dikumpulkan dengan program data collection.")
    exit()

print(f"\nTotal kelas: {len(labels)}")

# ============================================================
# 2. Build sequence (sliding window) untuk semua kelas
# ============================================================
no_of_timesteps = 20   # sliding window length
X, y = [], []

print("\n[2/6] Building sequences...")

label_to_id = {lbl: idx for idx, lbl in enumerate(labels)}
# contoh: {'neutral':0, 'A':1, 'B':2, ... 'Z':26}

def build_seq(df, label_id):
    data = df.values
    n = len(data)
    for i in range(no_of_timesteps, n):
        X.append(data[i-no_of_timesteps:i, :])
        y.append(label_id)

for lbl, df in dfs.items():
    lbl_id = label_to_id[lbl]
    build_seq(df, lbl_id)
    print(f"   Kelas {lbl:7s} -> label {lbl_id:2d}, sequence: {len(df) - no_of_timesteps}")

X = np.array(X)
y = np.array(y)

print(f"\n✅ X shape: {X.shape}")   # (samples, timesteps, features)
print(f"✅ y shape: {y.shape}")     # (samples,)

# Safety check
if X.shape[0] == 0:
    print("\n❌ ERROR: Tidak ada sequence yang terbentuk. Cek jumlah frame / no_of_timesteps.")
    exit()

# ============================================================
# 3. Split train-test
# ============================================================
print("\n[3/6] Splitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    shuffle=True,
    stratify=y,
    random_state=42
)

print(f"✅ Train: {len(X_train)} samples")
print(f"✅ Test : {len(X_test)} samples")

# ============================================================
# 4. Build LSTM model (27 kelas)
# ============================================================
print("\n[4/6] Building LSTM model...")

num_classes = len(labels)  # 27

model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    BatchNormalization(),
    Dropout(0.3),

    LSTM(64, return_sequences=True),
    BatchNormalization(),
    Dropout(0.3),

    LSTM(32),
    BatchNormalization(),
    Dropout(0.2),

    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("✅ Model compiled")
model.summary()

# ============================================================
# 5. Train dengan callbacks
# ============================================================
print("\n[5/6] Training model...")

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# ============================================================
# 6. Evaluate
# ============================================================
print("\n[6/6] Evaluating model...")
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

print(f"\n{'='*60}")
print("📊 HASIL TRAINING:")
print(f"{'='*60}")
print(f"  Test Loss     : {loss:.4f}")
print(f"  Test Accuracy : {accuracy*100:.2f}%")
print(f"{'='*60}")

# ============================================================
# 7. Save model + label map
# ============================================================
print("\n💾 Saving model & labels...")

model.save('gesture_model_az.h5')

# label_map: id -> label (kebalikan dari label_to_id)
label_map = {idx: lbl for lbl, idx in label_to_id.items()}

with open('gesture_labels_az.pkl', 'wb') as f:
    pickle.dump(label_map, f)

print("✅ gesture_model_az.h5 saved")
print("✅ gesture_labels_az.pkl saved")

print("\n" + "="*60)
print("🎉 TRAINING SELESAI (neutral + A-Z)!")
print("="*60)
print("📌 Langkah berikutnya:")
print("  1. Sesuaikan script live testing supaya load gesture_model_az.h5")
print("  2. Gunakan label_map baru dari gesture_labels_az.pkl")
print("="*60)
