import os
import random
import pandas as pd
import wave
import contextlib
import stat
import time

def create_pair_csv(root_dir, output_csv, pos_count, neg_count):
    """폴더 구조를 분석하여 Positive/Negative 쌍을 생성하고 CSV로 저장"""
    folders = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    
    folder_to_files = {}
    for folder in folders:
        wavs = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.wav')]
        if len(wavs) >= 2:
            folder_to_files[folder] = wavs

    all_pairs_set = set()
    positive_pairs = []
    negative_pairs = []

    # Positive pairs
    while len(positive_pairs) < pos_count:
        folder = random.choice(list(folder_to_files.keys()))
        f1, f2 = random.sample(folder_to_files[folder], 2)
        key = tuple(sorted((f1, f2)))
        if key not in all_pairs_set:
            all_pairs_set.add(key)
            positive_pairs.append((f1, f2, 1))

    # Negative pairs
    folder_keys = list(folder_to_files.keys())
    while len(negative_pairs) < neg_count:
        folder1, folder2 = random.sample(folder_keys, 2)
        f1, f2 = random.choice(folder_to_files[folder1]), random.choice(folder_to_files[folder2])
        key = tuple(sorted((f1, f2)))
        if key not in all_pairs_set:
            all_pairs_set.add(key)
            negative_pairs.append((f1, f2, 0))

    all_pairs = positive_pairs + negative_pairs
    random.shuffle(all_pairs)
    df = pd.DataFrame(all_pairs, columns=['f1', 'f2', 'label'])
    df.to_csv(output_csv, index=False)
    print(f"✅ {output_csv} 생성 완료: {len(df)}쌍")

def cleanup_wav_files(root_folder, min_sec=5, max_sec=20):
    """오디오 길이 필터링 및 삭제"""
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.lower().endswith(".wav"):
                file_path = os.path.join(dirpath, filename)
                try:
                    with contextlib.closing(wave.open(file_path, 'r')) as f:
                        duration = f.getnframes() / float(f.getframerate())
                    if duration < min_sec or duration > max_sec:
                        if not os.access(file_path, os.W_OK): os.chmod(file_path, stat.S_IWRITE)
                        os.remove(file_path)
                except Exception as e: print(f"Error: {e}")