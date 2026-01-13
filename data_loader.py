import numpy as np
import tensorflow as tf
import pandas as pd

def load_wav(filepath, sample_rate=16000, duration=10):
    """파일 하나를 읽어서 길이를 맞춘 뒤 numpy 배열로 반환"""
    num_samples = sample_rate * duration
    audio = tf.audio.decode_wav(tf.io.read_file(filepath), desired_channels=1).audio
    audio = tf.squeeze(audio, axis=-1)
    
    curr_samples = tf.shape(audio)[0]
    audio = tf.cond(
        curr_samples < num_samples,
        lambda: tf.pad(audio, [[0, num_samples - curr_samples]]),
        lambda: audio[:num_samples]
    )
    return audio.numpy()

class CachedWavPairDataset(tf.keras.utils.Sequence):
    """메모리 캐싱 기능이 포함된 데이터 로더 클래스"""
    def __init__(self, csv_path, batch_size=32, shuffle=True, sample_rate=16000, duration=10):
        self.df = pd.read_csv(csv_path)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.df))
        self.sample_rate = sample_rate
        self.duration = duration
        
        # 데이터 캐싱 (학습 속도 향상의 핵심)
        self.wav_cache = {}
        all_files = set(self.df['f1']).union(set(self.df['f2']))
        print(f"🚀 {len(all_files)}개의 오디오 파일을 메모리에 캐싱 중...")
        for fpath in all_files:
            self.wav_cache[fpath] = load_wav(fpath, self.sample_rate, self.duration)
        
        self.on_epoch_end()
    
    def __len__(self):
        return len(self.df) // self.batch_size
    
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch = self.df.iloc[batch_indices]
        
        X, y = [], []
        for _, row in batch.iterrows():
            f1_audio = self.wav_cache[row['f1']]
            f2_audio = self.wav_cache[row['f2']]
            # 두 오디오를 채널 방향으로 합침 (samples, 2)
            X.append(np.stack([f1_audio, f2_audio], axis=-1))
            y.append(row['label'])
            
        return np.stack(X), np.array(y).astype(np.float32)
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
            
    def clear_cache(self):
        """메모리 해제가 필요할 때 사용"""
        self.wav_cache.clear()
        import gc
        gc.collect()