# Agent.md

이 문서는 앞으로 이 저장소에서 작업하는 에이전트가 빠르게 맥락을 이해하고, 일관된 방식으로 구현할 수 있도록 돕는 작업 가이드입니다.

---

## 1. 프로젝트 한줄 요약

이 프로젝트는 `포켓몬 이미지 + 음성 멀티모달 분류 시스템`이며, 먼저 `SSW60` 논문 데이터셋으로 파이프라인을 검증한 뒤, 이후 포켓몬 전용 데이터셋으로 확장하는 구조입니다.

기준 문서

- [README.md](/C:/Users/user/AppData/Local/Temp/Free/Pokemon-Recognition-System/README.md)
- [주제.txt](/C:/Users/user/AppData/Local/Temp/Free/Pokemon-Recognition-System/주제.txt)
- [Reference Paper.pdf](</C:/Users/user/AppData/Local/Temp/Free/Pokemon-Recognition-System/Reference Paper.pdf>)

---

## 2. 현재 프로젝트 상태

- `Engine/IModel.py`, `ImageModel.py`, `AudioModel.py`, `LateFusionModel.py`, `MidFusionModel.py`, `ScoreFusionModel.py`가 이미 구현되어 있다.
- `Config/` 아래 모델별 기본 설정 파일 5개가 이미 생성되어 있다.
- `Data/SSW60/processed`는 `train/val/test/<label>/<sample_id>` 구조로 정리되어 있다.
- `scripts/prepare_ssw60_processed.py`로 SSW60 원본을 현재 규격으로 재생성할 수 있다.
- 기본 명령은 `python scripts\prepare_ssw60_processed.py`이고, 네트워크 없이 로컬 raw 또는 로컬 아카이브만 쓸 때는 `--skip-download`를 붙인다.
- `Data/SSW60`는 논문 기반 1차 검증용이다.
- `Data/Pokemon`은 최종 포켓몬 데이터셋 저장 위치다.
- `ImageModel`, `AudioModel`, `ScoreFusionModel`은 smoke dataset 기준 `Train`, `Load`, `Inference` 검증이 끝나 있다.
- `LateFusionModel`, `MidFusionModel`은 실제 샘플 기준 forward 검증이 끝나 있다.
- `Application` GUI는 아직 본격 구현 전이다.

즉, 현재는 "엔진 초기 구현 완료 이후 GUI 연동과 실험 고도화로 넘어가는 단계"로 보면 된다.

---

## 3. 핵심 목표

에이전트는 앞으로 아래 흐름을 기준으로 작업하면 된다.

1. SSW60 데이터셋으로 데이터 로딩, 전처리, 학습, 평가, 추론 파이프라인 검증
2. 이미지 단일 모달 모델 구현
3. 음성 단일 모달 모델 구현
4. 멀티모달 퓨전 3종 구현
5. 학습 GUI 구현
6. 파일 기반 테스트 GUI 구현
7. 실시간 테스트 GUI 구현
8. 포켓몬 데이터셋 적용

---

## 4. 모델 선택 기준

논문 전체를 그대로 재현하는 프로젝트가 아니라, 성능이 좋았던 대표 모델만 골라 구현하는 방향이다.

- 이미지 모델: `ViT-B/16`
- 음성 모델: `AST-style Audio Encoder (log-mel spectrogram + ViT-B/16)`
- 퓨전 방식
  - `MBT-inspired` mid fusion
  - `Late Fusion`
  - `Score Fusion`

추가 모델을 넣더라도 위 2개와 3개 퓨전 방식이 먼저다.

---

## 5. 아키텍처 원칙

### 5.1 가장 중요한 규칙

`Application`은 UI만 담당하고, 핵심 로직은 모두 `Engine`에 둔다.

### 5.2 Application 책임

- 윈도우/위젯 구성
- 사용자 입력 처리
- 진행도/로그/차트 표시
- 모델 결과 표시
- Engine 모델 클래스 호출

### 5.3 Engine 책임

- 데이터 전처리
- 데이터셋 로딩
- 모델 생성
- 학습 루프
- 평가
- 추론
- 퓨전 로직
- 체크포인트 저장/로딩
- 실시간 처리 서비스

현재 결정된 구현 방식

- 엔진 쪽은 파일을 잘게 나누지 않는다.
- `IModel.py`, `ImageModel.py`, `AudioModel.py`, `LateFusionModel.py`, `MidFusionModel.py`, `ScoreFusionModel.py`만 만든다.
- 각 파일 내부에 해당 클래스가 쓰는 전처리, 학습, 평가, 추론, 결과 저장 로직을 함께 넣는다.

### 5.4 금지 방향

가능하면 아래 구조는 피한다.

- GUI 코드 안에서 모델 직접 생성
- GUI 코드 안에서 학습 루프 실행
- UI 이벤트 핸들러 내부에 전처리/추론 핵심 로직 중복 작성
- 경로 규약을 여러 파일에서 제각각 해석하는 구조

---

## 6. 권장 디렉터리 구조

구현 시 아래 구조를 기본안으로 따른다.

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
```

현재 기준으로는 엔진 구현 파일을 추가로 잘게 분리하지 않는 방향을 우선한다.

---

## 7. 표준 데이터 규약

학습 프로그램은 사용자가 데이터셋 루트 폴더만 선택하면 되도록 설계한다.  
따라서 엔진은 아래 구조를 기본 학습 입력 규약으로 취급한다.

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

원본 데이터 구조가 다르다면 `scripts/` 또는 각 모델 파일 내부 변환 로직에서 위 표준 구조로 맞춘 뒤 학습에 사용한다.

---

## 8. 개발 환경 가정

현재 작업 환경 기준 가정

- OS: Windows 11 x64
- Shell: PowerShell
- Python: 3.10 이상
- 문자 인코딩: UTF-8
- 우선 가속 환경: NVIDIA GPU + CUDA
- CPU fallback도 가능해야 함

예상 주요 라이브러리

- PyTorch
- torchvision
- torchaudio
- PySide6
- OpenCV
- librosa
- soundfile
- scikit-learn
- matplotlib
- seaborn
- pyqtgraph

주의사항

- PowerShell에서는 한글 출력이 깨져 보일 수 있으므로, 파일 내용 검증은 필요 시 Python으로 UTF-8 재확인한다.
- 파일 저장은 가능하면 UTF-8 기준으로 유지한다.

---

## 9. 구현 우선순위 참고

아래 순서는 엔진을 처음 구현할 때 가장 안전한 흐름이다. 현재 저장소는 이 순서를 대부분 반영한 상태이며, 다음 우선순위는 GUI 연결과 전체 데이터셋 기준 재검증이다.

코드가 없을 때는 아래 순서가 가장 안전하다.

1. 공통 설정 파일 구조 결정
2. `IModel` 인터페이스 정의
3. `ImageModel` 구현
4. `AudioModel` 구현
5. 단일 모달 학습/평가/추론 검증
6. `LateFusionModel` 구현
7. `MidFusionModel` 구현
8. `ScoreFusionModel` 구현
9. 체크포인트/로그 저장 규약 정리
10. GUI 연결

GUI를 먼저 크게 만들기보다, 엔진을 먼저 CLI나 간단한 스크립트 기준으로 검증하고 붙이는 편이 좋다.

---

## 10. 파일 생성/수정 시 작업 원칙

### 10.1 설정 분리

- 하이퍼파라미터, 경로, 실행 모드는 코드에 박아두지 말고 프로젝트 루트의 `Config/`로 분리한다.

### 10.2 경로 처리

- 하드코딩한 절대 경로를 코드에 넣지 않는다.
- 프로젝트 루트 기준 상대 경로 또는 설정 파일 기반으로 처리한다.

### 10.3 실험 재현성

- seed 설정
- checkpoint 저장
- 학습 로그 기록
- 평가 결과 저장

위 4가지는 초반부터 같이 설계하는 편이 좋다.

### 10.4 시각화 산출물

- 그래프, confusion matrix, 리포트는 `artifacts/figures` 또는 `artifacts/reports`에 저장한다.

### 10.5 엔진 호출 방식

현재는 별도 서비스 레이어를 만들지 않는다.

- GUI는 `Engine`의 모델 클래스들을 직접 사용한다.
- 단, GUI 코드 안에서 학습/평가/추론 핵심 로직을 다시 작성하지 말고 `Train`, `Load`, `Inference` 인터페이스를 통해 호출한다.
- 공통 유틸이 필요하더라도 우선은 각 모델 파일 내부 private 함수나 내부 클래스로 정리한다.

---

## 11. GUI 설계 기준

이 프로젝트에서 GUI는 총 3종이다.

### 11.1 Trainer GUI

- 데이터셋 루트 선택
- 학습 모드 선택
- 퓨전 모드 선택
- 학습 시작/중지
- 로그 표시
- 진행도 표시
- 성능 지표 표시

### 11.2 File Test GUI

- 모델 선택
- 이미지 파일 선택
- 음성 파일 선택
- 단일/멀티모달 추론
- Top-k 결과 표시

### 11.3 Realtime Test GUI

- 웹캠 프리뷰
- 마이크 버퍼 입력
- 실시간 추론
- 실시간 결과 갱신

UI 코드에서는 가능하면 화면 상태와 사용자 입력 처리만 맡고, 모델 실행은 `Engine`의 모델 클래스 호출로 넘긴다.

---

## 12. 현재 핵심 구현 파일들

현재 기준으로 핵심 구현 파일과 폴더는 아래와 같다.

- `Engine/IModel.py`
- `Engine/ImageModel.py`
- `Engine/AudioModel.py`
- `Engine/LateFusionModel.py`
- `Engine/MidFusionModel.py`
- `Engine/ScoreFusionModel.py`
- `Config/`
- `scripts/prepare_ssw60_processed.py`

---

## 13. 에이전트가 기억해야 할 컨텍스트

- 이 프로젝트는 `포켓몬 과제형 + 논문 기반 응용`이다.
- 논문 구조를 참고하지만, 저장소 목적은 논문 재현 그 자체가 아니다.
- 1차 목표는 SSW60로 엔진을 검증하는 것이다.
- 최종 목표는 포켓몬 데이터셋 기반 GUI 시스템 완성이다.
- 현재는 엔진 골격 자체보다 GUI 연동, 실험 반복, 데이터셋 확장이 더 중요한 단계다.

---

## 14. 다음 액션 추천

이 문서를 읽은 다음 에이전트는 보통 아래 중 하나로 바로 이어가면 된다.

1. `Application`에서 `Engine` 모델 클래스를 직접 호출하도록 GUI를 연결
2. `LateFusionModel`, `MidFusionModel`까지 포함한 전체 학습 smoke 검증 보강
3. `Config/` 하이퍼파라미터를 조정하며 SSW60 기준 실험 반복
4. `Data/Pokemon` 원천 데이터를 현재 표준 구조로 변환하는 스크립트와 규격 정리
