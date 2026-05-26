# Pokemon Recognition System

포켓몬의 이미지와 음성을 함께 사용해 클래스를 분류하는 멀티모달 인식 시스템 프로젝트입니다.  
프로젝트 성격은 `포켓몬 과제형 프로젝트 + 논문 기반 응용`이며, 초기 단계에서는 논문 데이터셋(SSW60)으로 파이프라인을 검증하고, 이후 포켓몬 전용 데이터셋으로 확장하는 것을 목표로 합니다.
현재 저장소 기준으로는 `scripts/prepare_ssw60_processed.py`, `Engine` 6개 모델 파일, `Config` 기본 설정 파일, 학습 GUI, 파일 테스트 GUI가 준비되어 있으며, `Data`와 `artifacts`는 첫 실행 시 자동 생성됩니다.

기준 논문: `Exploring Fine-Grained Audiovisual Categorization with the SSW60 Dataset`

---

## Quick Guide

처음 클론한 뒤 바로 실행해볼 때의 최소 순서입니다.  
`Data/`와 `artifacts/`는 Git 추적 대상에서 제외되어 있으며, 아래 과정에서 자동으로 생성됩니다.

### 0. 환경 준비

권장 버전: `Python 3.10+`

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

- `requirements.txt`에는 현재 코드에서 실제로 사용하는 런타임 패키지만 넣어두었습니다.
- `Config/*.json`의 기본값은 `"device": "cuda"`입니다. GPU가 없으면 학습/추론 전에 `"cpu"`로 바꿔서 실행하세요.
- CUDA용 PyTorch가 필요하면 `torch`, `torchvision`, `torchaudio`는 PyTorch 공식 설치 가이드에 맞춰 다시 설치하는 편이 안전합니다.

### 1. SSW60 데이터 준비

fresh clone 상태에서는 아래 스크립트를 그대로 실행하면 됩니다.

```powershell
python scripts\prepare_ssw60_processed.py
```

이 스크립트는 다음을 수행합니다.

- SSW60 원본 아카이브를 다운로드한다.
- `Data\SSW60\raw`를 만든다.
- `Data\SSW60\processed` 표준 구조를 만든다.
- `train`, `val`, `test` 폴더와 `labels.json`을 생성한다.

이미 raw 데이터나 로컬 아카이브가 있을 때만 `--skip-download` 옵션을 사용하세요.

### 2. 학습 실행

학습 GUI는 아래 명령으로 실행합니다.

```powershell
python -m Application.trainer_gui.main
```

기본 흐름:

1. 데이터셋 폴더로 `Data\SSW60\processed`를 선택한다.
2. 모델을 고른다.
   `ImageModel`, `AudioModel`, `LateFusionModel`, `MidFusionModel`, `ScoreFusionModel`
3. `학습 대기열 추가`를 눌러 학습을 시작한다.
4. 결과는 `artifacts\training_runs\...` 아래에 저장된다.

학습 결과에서 주로 보는 파일:

- `checkpoints\*_best.pt`: 최고 성능 체크포인트
- `checkpoints\*_last.pt`: 마지막 체크포인트
- `logs\training_summary.json`
- `logs\training_history.json`
- `metrics\test_metrics.json`

### 3. 파일 기반 테스트

파일 테스트 GUI는 아래 명령으로 실행합니다.

```powershell
python -m Application.file_test_gui.main
```

기본 흐름:

1. 학습 결과 폴더의 `checkpoints\*_best.pt`를 선택한다.
2. 모델 종류에 맞는 입력 파일을 고른다.
   - `ImageModel`: 이미지 파일만 선택
   - `AudioModel`: 오디오 파일만 선택
   - 멀티모달 모델 3종: 이미지와 오디오를 모두 선택
3. `파일 기반 추론 실행`을 누른다.
4. 결과는 `artifacts\file_test_runs\...` 아래에 저장된다.

가장 쉽게 테스트하는 방법은 `Data\SSW60\processed\test\<label>\<sample_id>\` 안에 있는 `image.png`, `audio.wav`를 그대로 사용하는 것입니다.

파일 테스트 결과에서 주로 보는 파일:

- `predictions\inference_predictions.json`
- `logs\inference_summary.json`
- `metrics\inference_metrics.json`

### 4. 실시간 추론 실행

실시간 추론 GUI는 아래 명령으로 실행합니다.

```powershell
python -m Application.realtime_inference_gui.main
```

기본 흐름:

1. 학습 GUI가 만든 `artifacts\training_runs\...\checkpoints\*_best.pt` 또는 `*_last.pt`를 선택한다.
2. 모델 종류가 자동으로 맞춰지지 않으면 직접 선택한다.
3. 카메라 번호, 추론 간격, 오디오 윈도우 길이를 조정한다.
4. `실시간 추론 시작`을 누르면 카메라/마이크 입력을 계속 샘플링해 예측, 신뢰도, Top-K 결과를 갱신한다.

실시간 앱은 기존 엔진의 `Load()`와 `Inference()`를 재사용합니다. 따라서 각 추론 tick은 `artifacts\realtime_runs\...` 아래에 `image.png`, `audio.wav`, `predictions\inference_predictions.json` 형태로 저장됩니다.

---

## 1. 프로젝트 목적

- 이미지 단일 모달 분류와 음성 단일 모달 분류를 각각 구현한다.
- 이미지 + 음성 멀티모달 분류가 단일 모달 대비 어떤 성능 차이를 보이는지 확인한다.
- 논문에서 사용한 대표적인 퓨전 방식 3가지를 동일한 프로젝트 구조 안에서 비교한다.
- 최종적으로는 학습 프로그램, 파일 기반 테스트 프로그램, 실시간 테스트 프로그램까지 포함한 데스크톱 GUI 시스템을 완성한다.

---

## 2. 프로젝트 진행 방향

### 2.1 1차 단계: 논문 기반 검증

- `Data/SSW60`를 사용해 전체 학습/평가/추론 파이프라인을 먼저 검증한다.
- 이미지 모델 1종, 음성 모델 1종, 퓨전 방식 3종이 정상적으로 동작하는지 확인한다.
- 논문 실험과 유사한 방식으로 다음 구성을 비교한다.
  - Image only
  - Audio only
  - Image + Audio

### 2.2 2차 단계: 포켓몬 데이터셋 적용

- 포켓몬 이미지와 포켓몬 음성(울음소리) 데이터를 자체 규격에 맞춰 구축한다.
- 1차 검증에서 완성된 엔진을 그대로 재사용해 포켓몬 분류 모델을 학습한다.
- 최종적으로 파일 기반 테스트와 웹캠/마이크 기반 실시간 테스트까지 연결한다.

---

## 3. 핵심 구현 범위

### 3.1 선택 모델

이 프로젝트는 논문에 나온 후보를 모두 구현하는 것이 아니라, 성능이 좋았던 계열만 추려서 최소 구성으로 구현한다.

- 이미지 모델: `ViT-B/16`
  - 논문 멀티모달 실험에서 사용된 시각 백본 계열
  - 포켓몬 이미지 분류의 기본 시각 인코더로 사용

- 음성 모델: `AST-style Audio Encoder`
  - log-mel spectrogram을 입력으로 받는 `ViT-B/16` 기반 음성 분류기
  - 별도 AST 패키지 재현 대신 AST 계열 접근을 현재 저장소 구조에 맞춰 단일 파일로 구현
  - 논문에서 음성 쪽 성능이 좋았던 Transformer 계열 접근을 유지

### 3.2 퓨전 방식

논문 기준 3가지 방식을 모두 비교 대상으로 둔다.

- `Mid Fusion`: MBT-inspired bottleneck transformer fusion
- `Late Fusion`: 이미지/음성 임베딩 결합 후 분류
- `Score Fusion`: 이미지 분류 점수와 음성 분류 점수의 가중 합

### 3.3 평가 지표

- Accuracy
- Macro F1-score
- Precision / Recall / F1
- Confusion Matrix
- 학습 곡선(Loss / Accuracy)

---

## 4. 방법론

### 4.1 이미지 처리

- 입력 이미지를 표준 크기로 정규화
- 학습 시 랜덤 크롭, 좌우 반전, 회전, 블러, 밝기 변화 등의 증강 적용
- 단일 이미지 분류와 멀티모달 시각 입력 양쪽에 공통 사용

### 4.2 음성 처리

- 음성 파일을 단일 샘플링 레이트로 통일
- 멜 스펙트로그램으로 변환 후 모델 입력으로 사용
- 시간 축 crop, 주파수 마스킹, 환경 노이즈 추가 등의 증강 적용
- 실시간 추론에서는 마이크 입력을 짧은 윈도우 단위로 잘라 동일한 전처리 적용

### 4.3 멀티모달 학습

- 동일 샘플에 대해 이미지와 음성을 함께 읽는다.
- 같은 데이터셋으로 다음 세 가지 실험을 반복한다.
  - 이미지 전용 학습
  - 음성 전용 학습
  - 이미지 + 음성 멀티모달 학습
- 멀티모달 학습에서는 MBT, Late Fusion, Score Fusion을 각각 분리된 실험으로 관리한다.

### 4.4 실험 흐름

1. SSW60 데이터셋으로 엔진 검증
2. 이미지/음성 단일 모달 성능 확인
3. 퓨전 3종 성능 비교
4. 포켓몬 데이터셋 구축 및 동일 파이프라인 적용
5. GUI 프로그램 3종 연결

---

## 5. 프로그램 구성

### 5.1 학습 프로그램

목적: 사용자가 데이터셋 루트 폴더만 선택하면 정해진 구조를 기준으로 바로 학습할 수 있는 GUI

주요 기능

- 데이터셋 루트 폴더 선택
- 학습 모드 선택
  - Image only
  - Audio only
  - Multimodal
- 퓨전 방식 선택
  - MBT
  - Late Fusion
  - Score Fusion
- 학습 시작 / 중단
- 학습 진행도 시각화
- 학습 로그 시각화
- 성능 지표 시각화
- 체크포인트 저장
- 최고 성능 모델 저장

### 5.2 파일 기반 테스트 프로그램

목적: 학습 완료 모델을 사용해 이미지 파일, 음성 파일, 또는 이미지+음성 쌍으로 1차 테스트를 수행하는 GUI

주요 기능

- 학습된 모델 선택
- 이미지 파일 선택
- 음성 파일 선택
- 단일/멀티모달 추론 실행
- 예측 클래스, 확률, Top-k 결과 표시

### 5.3 실시간 기반 테스트 프로그램

목적: 노트북 웹캠과 마이크를 이용해 최종 실시간 데모를 수행하는 GUI

주요 기능

- 웹캠 프리뷰
- 마이크 입력 수집
- 실시간 프레임/오디오 버퍼 추론
- 예측 결과 실시간 갱신
- 신뢰도 점수 표시
- 결과 로그 저장 옵션

---

## 6. 사용 기술

### 6.1 언어 및 기본 환경

- Python 3.10+

### 6.2 딥러닝 / 데이터 처리

- PyTorch
- torchvision
- torchaudio
- NumPy
- pandas
- scikit-learn

### 6.3 이미지 / 음성 처리

- OpenCV
- librosa
- soundfile

### 6.4 시각화

- Matplotlib
- seaborn
- pyqtgraph

### 6.5 GUI

- PySide6

---

## 7. 프레임워크 및 플랫폼

### 7.1 딥러닝 프레임워크

- `PyTorch` 기반으로 학습, 평가, 추론 엔진 구현
- 모델 교체와 실험 비교가 쉽도록 `IModel` 기반 클래스 구조로 설계

### 7.2 GUI 프레임워크

- `PySide6` 기반 데스크톱 GUI 채택
- 이유
  - Python과 결합이 자연스럽다.
  - 멀티 윈도우 구성에 적합하다.
  - 학습 상태, 차트, 로그, 실시간 화면 표시를 한 프로젝트 안에서 통합하기 쉽다.
  - 최종 목표인 `학습 프로그램 / 파일 테스트 / 실시간 테스트` 3개 앱 구조에 잘 맞는다.

### 7.3 목표 플랫폼

- 1차 개발 플랫폼: `Windows 11 x64`
- 실행 대상 플랫폼: `Windows 10/11 x64 노트북`
- 가속 환경: `NVIDIA GPU + CUDA` 우선
- CPU 환경도 동작 가능하도록 설계하되, 실시간 성능은 GPU 환경을 기준으로 최적화

---

## 8. 데이터 구조

이 프로젝트의 학습 프로그램은 사용자가 복잡한 설정 없이 `데이터셋 루트 폴더`만 선택하도록 설계한다.  
따라서 학습용 데이터는 반드시 아래의 표준 구조를 따르는 것을 전제로 한다.

### 8.1 표준 학습 데이터 구조

```text
<dataset_root>/
  train/
    <class_name>/
      <sample_id>/
        image.png
        audio.wav
        meta.json
  val/
    <class_name>/
      <sample_id>/
        image.png
        audio.wav
        meta.json
  test/
    <class_name>/
      <sample_id>/
        image.png
        audio.wav
        meta.json
  labels.json
```

### 8.2 구조 설명

- `class_name`: 포켓몬 이름 또는 클래스 ID
- `sample_id`: 한 쌍의 멀티모달 샘플 단위
- `image.png`: 해당 샘플의 이미지
- `audio.wav`: 해당 샘플의 음성
- `meta.json`: 선택 사항, 수집 정보/증강 이력/원본 경로 등을 저장
- `labels.json`: 클래스 인덱스 매핑 정보

### 8.3 설계 의도

- 멀티모달 학습에서는 이미지와 음성이 같은 `sample_id` 아래에 있어야 한다.
- Image only / Audio only 실험에서는 같은 구조를 그대로 사용하고, 필요한 파일만 사용한다.
- 원본 데이터셋 구조가 다르더라도 `Engine` 내부 전처리 단계에서 위 표준 구조로 변환한 뒤 학습 프로그램이 읽도록 한다.

### 8.4 `scripts/prepare_ssw60_processed.py` 사용법

`SSW60` 원본을 이 프로젝트 표준 구조로 맞출 때는 [scripts/prepare_ssw60_processed.py](/C:/Users/user/AppData/Local/Temp/Free/Pokemon-Recognition-System/scripts/prepare_ssw60_processed.py:20)를 사용한다.

스크립트가 한 번에 처리하는 흐름:

- raw 데이터가 없으면 공식 `SSW60` tar.gz를 다운로드한다.
- 아카이브 MD5를 확인한다.
- `Data/SSW60/raw` 아래로 압축을 푼다.
- 비디오 자산을 제거한다.
- `Data/SSW60/processed` 표준 구조를 생성한다.
- paired `val`이 없으면 `train` 일부를 떼어 `val`로 만든다.

기본 사용:

```powershell
python scripts\prepare_ssw60_processed.py
```

이미 raw 또는 로컬 아카이브가 있고 네트워크 없이 처리할 때:

```powershell
python scripts\prepare_ssw60_processed.py --skip-download
```

자주 쓰는 옵션:

- `--raw-root`: raw 데이터셋 위치 지정
- `--processed-root`: processed 출력 위치 지정
- `--archive-path`: tar.gz 파일 경로 지정
- `--val-ratio`: `val` 파생 비율 지정
- `--keep-archive`: 압축 해제 후 tar.gz 유지
- `--keep-videos`: 비디오 파일과 `video_ml.csv` 유지
- `--skip-md5-check`: MD5 검증 생략
- `--force-download`: 기존 아카이브가 있어도 다시 다운로드
- `--download-url`: 다른 다운로드 URL 사용

예시:

```powershell
python scripts\prepare_ssw60_processed.py `
  --raw-root Data\SSW60\raw `
  --processed-root Data\SSW60\processed `
  --archive-path Data\SSW60\ssw60.tar.gz `
  --val-ratio 0.1
```

---

## 9. 권장 폴더 구조

현재 저장소의 상위 구조는 `Application`, `Data`, `Engine` 중심으로 유지하되, 엔진 구현은 아래처럼 **클래스별 단일 파일 구조**를 기준으로 진행한다.

```text
Pokemon-Recognition-System/
  Application/
    common/
    trainer_gui/
    file_test_gui/
    realtime_test_gui/
  Data/
    SSW60/
      raw/
      processed/
    Pokemon/
      raw/
      processed/
  Engine/
    IModel.py
    ImageModel.py
    AudioModel.py
    LateFusionModel.py
    MidFusionModel.py
    ScoreFusionModel.py
  artifacts/
    checkpoints/
    figures/
    reports/
    exports/
  Config/
  logs/
  scripts/
  tests/
  README.md
```

---

## 10. 소스 구조 설계 원칙

### 10.1 Application

`Application`은 GUI 껍데기만 담당한다.

역할

- 윈도우/위젯/화면 전환
- 사용자 입력 수집
- 학습 상태 및 추론 결과 표시
- 차트와 로그 출력

넣지 말아야 할 것

- 데이터 전처리 로직
- 모델 정의
- 학습 루프
- 추론 핵심 로직
- 퓨전 알고리즘 구현

### 10.2 Engine

`Engine`은 프로젝트의 모든 핵심 로직을 담당한다.

역할

- 데이터셋 로딩 및 구조 검증
- 이미지/음성 전처리
- 모델 생성 및 가중치 로딩
- 학습/검증/테스트 루프
- 퓨전 방식 구현
- 성능 평가
- 파일 기반 추론
- 실시간 추론 서비스

현재 구현 정책:

- 엔진 쪽은 파일을 잘게 나누지 않는다.
- `IModel.py`, `ImageModel.py`, `AudioModel.py`, `LateFusionModel.py`, `MidFusionModel.py`, `ScoreFusionModel.py`만 만든다.
- 각 파일 내부에 해당 클래스가 사용하는 전처리, 학습, 평가, 추론, 저장 로직을 함께 포함한다.

### 10.3 Application-Engine 연결 방식

- 현재 버전에서는 별도 `Engine/services` 레이어를 두지 않는다.
- `Application`은 필요한 모델 클래스를 `Engine`에서 직접 불러와 사용한다.
- 단, GUI 코드 안에 학습/추론 핵심 로직을 다시 작성하지 않고, 실제 실행은 각 모델 클래스의 `Train`, `Load`, `Inference`를 통해 수행한다.

예시

- `ImageModel`
- `AudioModel`
- `LateFusionModel`
- `MidFusionModel`
- `ScoreFusionModel`

### 10.4 모델 인터페이스 규약

모델 계층은 공통 인터페이스 `IModel`을 기준으로 구성한다.

구성 대상

- `IModel`: 모든 분류 모델의 공통 인터페이스
- `ImageModel`: `IModel`을 상속받는 이미지 분류 모델
- `AudioModel`: `IModel`을 상속받는 음성 분류 모델
- `LateFusionModel`: `IModel`을 상속받는 이미지+음성 분류 모델
- `MidFusionModel`: `IModel`을 상속받는 이미지+음성 분류 모델
- `ScoreFusionModel`: `IModel`을 상속받는 이미지+음성 분류 모델

모든 모델은 아래 인터페이스를 반드시 제공한다.

```text
Train(DataSetPath, ResultSavePath)
Load(NetworkFilePath)
Inference(DataPath, ResultSavePath)
```

각 인터페이스의 의미

- `Train(DataSetPath, ResultSavePath)`
  - 데이터셋 폴더 경로를 입력받아 학습을 수행한다.
  - 학습 로그, 학습 완료 후 테스트 지표, 네트워크 파일을 `ResultSavePath`에 저장한다.
- `Load(NetworkFilePath)`
  - 저장된 네트워크 파일 경로를 입력받아 해당 신경망을 로드한다.
- `Inference(DataPath, ResultSavePath)`
  - 데이터가 들어있는 폴더 경로를 입력받아 추론을 수행한다.
  - 추론 결과를 `ResultSavePath`에 저장한다.

모델별 역할

- `ImageModel`은 이미지 단일 모달 분류를 담당한다.
- `AudioModel`은 음성 단일 모달 분류를 담당한다.
- `LateFusionModel`은 이미지 특징과 음성 특징을 후단에서 결합해 분류한다.
- `MidFusionModel`은 이미지와 음성의 중간 표현을 결합해 분류한다.
- `ScoreFusionModel`은 이미지 점수와 음성 점수를 결합해 분류한다.

파일 구성 원칙

- 클래스마다 `.py` 파일 하나씩만 둔다.
- 별도 `trainer.py`, `metrics.py`, `service.py`, `dataset.py`, `pipeline.py` 파일은 현재 기준 만들지 않는다.
- 각 파일은 해당 클래스가 동작하는 데 필요한 내부 구현을 함께 포함하는 자기완결형 구조로 작성한다.
- 공통 실행 옵션은 코드 인자가 아니라 `Config/` 설정 파일을 통해 통제한다.

설정 파일 원칙

- 위 인터페이스는 경로 인자만 받는다.
- 하이퍼파라미터, 데이터 증강 여부, Dropout 정도, Early Stop 여부, Optimizer, Learning Rate, Batch Size 등 실행 파라미터는 메서드 인자로 받지 않는다.
- 위 파라미터는 모두 프로젝트 루트의 `Config/` 폴더 아래 설정 파일을 통해 결정한다.

---

## 11. Engine 파일 구성

### 11.1 `Engine/IModel.py`

- `Train`, `Load`, `Inference` 인터페이스 정의
- 모든 모델이 공통으로 따라야 할 계약 정의
- 필요 시 최소 공통 유틸 또는 공통 설정 로딩 진입점 포함

### 11.2 `Engine/ImageModel.py`

- 이미지 전처리
- 이미지 데이터 로딩
- 이미지 모델 생성
- 이미지 학습/평가/추론
- 이미지 결과 저장

### 11.3 `Engine/AudioModel.py`

- 음성 전처리
- 스펙트로그램 변환
- 음성 데이터 로딩
- 음성 모델 생성
- 음성 학습/평가/추론
- 음성 결과 저장

### 11.4 `Engine/LateFusionModel.py`

- 이미지+음성 로딩
- 이미지 특징 추출
- 음성 특징 추출
- Late Fusion 학습/평가/추론
- 결과 저장

### 11.5 `Engine/MidFusionModel.py`

- 이미지+음성 로딩
- 중간 표현 퓨전 구현
- Mid Fusion 학습/평가/추론
- 결과 저장

### 11.6 `Engine/ScoreFusionModel.py`

- 이미지 점수 산출
- 음성 점수 산출
- 점수 결합 규칙 구현
- Score Fusion 학습/평가/추론
- 결과 저장

### 11.7 공통 구현 위치 원칙

- 데이터셋 검증, 전처리, 체크포인트 저장, 지표 계산 같은 세부 구현도 별도 파일로 분리하지 않는다.
- 각 모델 파일 내부에서 private 함수 또는 내부 클래스로 정리한다.
- 중복보다 현재 프로젝트의 단순한 파일 구조 유지가 우선이다.

---

## 12. 구현 순서 참고

아래 순서는 엔진을 처음 올릴 때의 기준이다. 현재 저장소 기준으로는 `Config/` 기본 설정 파일과 `Engine` 6개 모델 파일이 이미 반영되어 있다.

현재 검증 상태:

- `ImageModel`, `AudioModel`, `ScoreFusionModel`은 소규모 smoke dataset 기준 `Train -> Load -> Inference`까지 확인했다.
- `LateFusionModel`, `MidFusionModel`은 실제 샘플 입력 기준 forward 검증을 마쳤다.
- 다음 우선순위는 GUI 연결, 전체 데이터셋 학습 검증, 포켓몬 데이터셋 확장이다.

1. `Config/` 설정 파일 구조 확정
2. `Engine/IModel.py` 인터페이스 정의
3. `Engine/ImageModel.py` 구현
4. `Engine/AudioModel.py` 구현
5. 단일 모달 학습/평가 파이프라인 검증
6. `Engine/LateFusionModel.py` 구현
7. `Engine/MidFusionModel.py` 구현
8. `Engine/ScoreFusionModel.py` 구현
9. 학습 GUI 연결
10. 파일 기반 테스트 GUI 연결
11. 포켓몬 데이터셋 구조 확정 및 데이터 구축
12. 포켓몬 데이터셋 재학습
13. 웹캠/마이크 기반 실시간 GUI 연결

---

## 13. 기대 결과

- 논문 구조를 응용한 포켓몬 멀티모달 분류 시스템 확보
- 단일 모달과 멀티모달 성능 비교 결과 확보
- 퓨전 방식 3종 비교 결과 확보
- 학습, 파일 테스트, 실시간 테스트를 모두 포함하는 GUI 기반 프로젝트 완성
