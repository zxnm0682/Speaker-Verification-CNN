# 음성쌍을 이용한 동일 화자 검증 CNN

오디오 샘플 쌍을 비교하여 동일인 여부를 판별하는 이진 분류 딥러닝 프로젝트입니다. 대규모 오디오 데이터를 효율적으로 처리하기 위한 메모리 캐싱 데이터 파이프라인을 활용했습니다.

프로젝트 구조

dataset_creator.py	원본 데이터를 스캔하여 학습용/테스트용 CSV 쌍을 생성하고, 오디오 길이를 기준으로 필터링 및 클리닝을 수행합니다.

dataloader.py	WAV 파일을 로드하고, 학습 속도를 높이기 위해 메모리에 데이터를 캐싱하는 CachedWavPairDataset 클래스를 포함합니다.

model.py	1D CNN 아키텍처 정의 및 컴파일, 학습(Fit), 콜백(EarlyStopping, ReduceLROnPlateau) 설정 등 학습 파이프라인을 담당합니다.

visualize.py	모델 성능 분석을 위한 Loss/Accuracy 그래프 시각화 및 Confusion Matrix 출력 함수를 포함합니다.

main.py	전체 워크플로우를 제어하는 엔트리 포인트입니다.

Model Architecture

본 프로젝트는 시계열 데이터인 오디오 파형(Waveform)을 직접 처리하기 위해 1D Convolutional Neural Network를 사용합니다.

Input: (160,000, 2) - 10초 분량의 16kHz 오디오 샘플 2개

Feature Extractor: 3계층의 Conv1D 레이어와 BatchNormalization, MaxPooling1D 적용

Classifier: GlobalAveragePooling1D를 거쳐 Dense 레이어와 Sigmoid 활성화 함수를 통해 최종 유사도 점수 출력

학습이 완료되면 visualize.py를 통해 다음과 같은 리포트를 생성합니다.

Learning Curves: Epoch에 따른 Loss 및 Accuracy 변화 확인 (과적합 모니터링)

Confusion Matrix: 모델이 'Same'과 'Different'를 얼마나 정확하게 구분하는지 수치화

본 프로젝트에 활용한 데이터셋은 AIHub의 '한국어 대학 강의 데이터'의 검증 데이터셋에서 서브셋을 만들어 구성했습니다.
만들어진 서브셋은 학습과 검증 데이터셋 각각 중복되지 않은 100명의 인당 100개 이상의 음성으로 구성되었습니다.(총 26,919개의 음성 데이터 활용)
