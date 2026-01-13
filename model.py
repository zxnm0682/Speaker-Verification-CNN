import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from data_loader import CachedWavPairDataset
import gc

def create_cnn_model(input_shape=(160000, 2)):
    """1D CNN 기반 오디오 쌍 비교 모델 아키텍처"""
    inputs = tf.keras.Input(shape=input_shape)
    
    x = tf.keras.layers.Conv1D(128, kernel_size=5, strides=2, activation='relu')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    x = tf.keras.layers.Conv1D(64, kernel_size=5, strides=2, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    x = tf.keras.layers.Conv1D(32, kernel_size=5, strides=2, activation='relu')(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs)

def train_model(train_csv, test_csv, batch_size=16, epochs=10):
    """학습 로직: 데이터 로딩, 학습, 메모리 정리 통합"""
    # 1. 데이터셋 생성
    train_data = CachedWavPairDataset(train_csv, batch_size=batch_size)
    test_data = CachedWavPairDataset(test_csv, batch_size=batch_size, shuffle=False)

    # 2. 모델 생성 및 컴파일
    model = create_cnn_model()
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # 3. 콜백 설정
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        verbose=1,
        min_lr=1e-6
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=7,
        restore_best_weights=True
    )

    # 4. 학습 실행
    history = model.fit(
        train_data, 
        validation_data=test_data, 
        epochs=epochs, 
        shuffle=True, 
        verbose=2, 
        callbacks=[reduce_lr, early_stopping]
    )

    # 5. 메모리 캐시된 WAV 삭제 및 정리
    del train_data.wav_cache
    del test_data.wav_cache
    gc.collect()

    return model, history