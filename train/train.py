import os
import multiprocessing as mp

# Fix macOS fork/mutex crash
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)  # <- safe multiprocessing start

    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    import matplotlib.pyplot as plt
    from tqdm.keras import TqdmCallback

    # Paths
    train_dir = "/Users/naba/Desktop/freelance/data_qr/train"
    val_dir = "/Users/naba/Desktop/freelance/data_qr/val"
    save_path = "/Users/naba/Desktop/freelance/data_qr/saved_models"

    os.makedirs(save_path, exist_ok=True)

    # Load datasets
    img_size = (128, 128)
    batch_size = 32

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="binary"
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="binary"
    )

    # Prefetching for performance
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    # Model
    model = keras.Sequential([
        layers.Rescaling(1./255, input_shape=(128, 128, 3)),

        layers.Conv2D(16, (3,3), activation='relu'),
        layers.MaxPooling2D(),

        layers.Conv2D(32, (3,3), activation='relu'),
        layers.MaxPooling2D(),

        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')  # binary classification
    ])

    # Compile with additional metrics
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )

    # Callbacks: Save best model
    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(save_path, "qr_model.h5"),
        save_best_only=True,
        monitor="val_accuracy",
        mode="max",
        verbose=1
    )

    # Train with tqdm callback
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=15,
        callbacks=[TqdmCallback(verbose=1), checkpoint_cb]
    )
