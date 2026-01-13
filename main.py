import os
from dataset_creator import create_pair_csv, cleanup_wav_files
from model import train_model
from visualize import plot_history, plot_cm
from data_loader import CachedWavPairDataset

# --------------------------
# 1. 경로 및 하이퍼파라미터 설정
# --------------------------
# 실제 데이터가 있는 상위 폴더 경로로 수정하세요.
TRAIN_ROOT_DIR = r'C:\workspace\dataset\train'
TEST_ROOT_DIR = r'C:\workspace\dataset\test'

# 생성할 CSV 파일명
TRAIN_CSV = 'train_pairs.csv'
TEST_CSV = 'test_pairs.csv'

# 학습 설정
BATCH_SIZE = 8
EPOCHS = 60

def main():
    # --------------------------
    # 2. 데이터 준비 (CSV 생성 및 클리닝)
    # --------------------------
    print("🧹 오디오 파일 길이 필터링 중...")
    cleanup_wav_files(TRAIN_ROOT_DIR)
    cleanup_wav_files(TEST_ROOT_DIR)

    print("📊 학습용/테스트용 쌍(Pair) 생성 중...")
    # 학습 데이터셋: Positive 12,000 / Negative 24,000 생성 예시
    create_pair_csv(TRAIN_ROOT_DIR, TRAIN_CSV, pos_count=12000, neg_count=24000)
    
    # 테스트 데이터셋: Positive 3,000 / Negative 7,000 생성 예시
    create_pair_csv(TEST_ROOT_DIR, TEST_CSV, pos_count=3000, neg_count=7000)

    # --------------------------
    # 3. 모델 학습 실행
    # --------------------------
    # train_model 내부에서 데이터 로딩(캐싱), 컴파일, 학습, 메모리 정리가 모두 수행됩니다.
    print(f"🚀 학습 시작... (Batch Size: {BATCH_SIZE}, Epochs: {EPOCHS})")
    model, history = train_model(
        train_csv=TRAIN_CSV, 
        test_csv=TEST_CSV, 
        batch_size=BATCH_SIZE, 
        epochs=EPOCHS
    )

    # --------------------------
    # 4. 시각화 (Loss & Accuracy)
    # --------------------------
    print("📈 학습 결과 시각화 중...")
    plot_history(history)

    # --------------------------
    # 5. Confusion Matrix 출력
    # --------------------------
    # train_model 함수 마지막에 캐시가 삭제되므로, 시각화를 위해 테스트 데이터를 다시 로드합니다.
    print("🔍 혼동 행렬(Confusion Matrix) 생성 중...")
    test_data_for_eval = CachedWavPairDataset(TEST_CSV, batch_size=BATCH_SIZE, shuffle=False)
    plot_cm(model, test_data_for_eval)
    
    # 최종 메모리 정리
    test_data_for_eval.clear_cache()
    print("✨ 모든 프로세스가 완료되었습니다.")

if __name__ == "__main__":
    main()