# RecBole AutoML 프로젝트

Kaggle 채용 추천 대회를 위한 RecBole + Ray Tune 기반 AutoML 프로젝트입니다.

---

## 1. 아나콘다 가상환경 생성

### 환경 생성 (처음 한 번만 실행)

```bash
# Python 3.10 환경 생성 (필수)
conda create -n recbole python=3.10 -y

# 환경 활성화
conda activate recbole

# 패키지 설치
pip install -r requirements.txt
```

**중요**: Python 3.10이 필수입니다. Ray Tune 2.20.0과의 호환성을 위해 다른 버전을 사용하지 마세요.

**이미 환경이 있다면** 환경 생성 단계는 건너뛰고 활성화부터 진행하세요.

---

## 2. 가상환경 활성화/비활성화

### 활성화
```bash
conda activate recbole
```

### 비활성화
```bash
conda deactivate
```

---

## 3. 가상환경 제거

```bash
# 환경 제거 (주의: 되돌릴 수 없습니다)
conda env remove -n recbole
```

---

## 4. 프로젝트 특징

### AutoML with Ray Tune
- **모델**: LightGCN (Graph Neural Network)
- **최적화 도구**: Ray Tune 2.20.0
- **스케줄러**: ASHA (Asynchronous Successive Halving Algorithm)
- **검색 알고리즘**: Optuna TPE (Tree-structured Parzen Estimator)
- **하이퍼파라미터 공간**:
  - `embedding_size`: [32, 64, 128, 256]
  - `n_layers`: [1, 2, 3, 4]
  - `reg_weight`: [1e-5, 1e-2] (log-uniform)
  - `learning_rate`: [1e-4, 1e-2] (log-uniform)

### 데이터셋 특성
- **사용자 수**: 8,482명
- **아이템 수**: 6,695개
- **상호작용 수**: 57,946건
- **희소성**: 99.90% (매우 희소)
- **평가 메트릭**: Recall@5 (주), NDCG@5, Recall@10

### 디바이스 자동 선택
자동으로 최적의 디바이스를 선택합니다:
1. CUDA (NVIDIA GPU)
2. MPS (Apple Silicon GPU)
3. CPU (폴백)

---

## 5. 데이터 경로

### 입력 데이터
```
dataset/apply_train.csv
```

**데이터 형식**:
- `resume_seq`: 사용자 ID
- `recruitment_seq`: 채용공고 ID
- 자동 변환: `user_id:token`, `item_id:token`, `rating:float` (1.0)

### RecBole 데이터셋
```
dataset/kaggle_recsys/kaggle_recsys.inter
```
스크립트 실행 시 자동 생성됩니다.

### 출력 데이터

#### Ray Tune 결과
```
ray_results/recbole_lightgcn_automl/
```
- 모든 trial 결과 저장
- 최적 하이퍼파라미터 로그
- 학습 곡선 및 메트릭

#### 제출 파일
```
outputs/{YYYY-MM-DD}/submit_LightGCN_RayTune_{YYYYMMDDhhmmss}.csv
```

**출력 예시**:
```
outputs/2025-11-20/submit_LightGCN_RayTune_20251120123045.csv
```

**Row-per-recommendation 형식**:
```csv
resume_seq,recruitment_seq
U05833,R03838
U05833,R02144
U05833,R01877
...
```

**참고**:
- `dataset/` 디렉토리는 수동으로 생성 필요
- `outputs/` 디렉토리는 스크립트 실행 시 자동 생성
- 제출 파일은 날짜별로 자동 분류됨

---

## 6. 프로젝트 구조

```
recbole_automl/
├── dataset/
│   ├── apply_train.csv          # 입력 데이터
│   └── kaggle_recsys/
│       └── kaggle_recsys.inter  # RecBole 형식 (자동 생성)
├── outputs/
│   └── {YYYY-MM-DD}/
│       └── submit_*.csv         # 제출 파일 (날짜별)
├── ray_results/
│   └── recbole_lightgcn_automl/ # Ray Tune 최적화 결과
├── auto_ml_raytune.py           # Python 스크립트 (추천)
├── auto_ml.ipynb                # Jupyter 노트북
├── requirements.txt             # 패키지 목록
├── CLAUDE.md                    # Claude Code 가이드
└── README.md
```

---

## 7. 버전 요구사항

**중요**: 아래 버전 제약을 반드시 지켜야 합니다.

### Python
- **필수**: Python 3.10

### 핵심 라이브러리
- `ray[tune]==2.20.0` (정확히 이 버전)
- `pyarrow>=14.0.0,<15.0.0` (Ray 2.20.0 호환)
- `pydantic>=2.0.0,<3.0.0`
- `optuna>=3.0.0`

### RecBole 및 의존성
- `recbole>=1.1.1`
- `numpy>=1.24.0,<2.0` (NumPy 2.0 미지원)
- `scipy>=1.10.0,<1.13.0` (SciPy 1.13+ 미지원)

### PyTorch
- `torch>=2.0.0`
- `torchvision>=0.15.0`
- `torchaudio>=2.0.0`

자세한 버전 정보는 `requirements.txt`를 참조하세요.

---

## 8. 문제 해결

### Ray Tune verbose 버그 (Python 3.12)
Python 3.12 환경에서 Ray 2.31.0+ 버전은 verbose 관련 버그가 있습니다.
**해결책**: Python 3.10 + Ray 2.20.0 사용 (이미 적용됨)

### PyArrow 호환성 에러
```
AttributeError: module 'pyarrow' has no attribute 'PyExtensionType'
```
**해결책**:
```bash
pip install 'pyarrow>=14.0.0,<15.0.0' --force-reinstall
```

### RecBole 의존성 에러

**NumPy 2.0 에러**:
```bash
pip install 'numpy>=1.24.0,<2.0'
```

**SciPy 1.13+ 에러**:
```bash
pip install 'scipy>=1.10.0,<1.13.0'
```

### GPU 디바이스 에러
MPS/CUDA 텐서 에러 발생 시, 코드에서 자동으로 디바이스를 감지하고 설정합니다.
절대 경로가 사용되므로 Ray Tune의 병렬 실행에서도 문제없이 작동합니다.

---

## 9. 참고 문서

- **CLAUDE.md**: Claude Code용 프로젝트 가이드
- **requirements.txt**: 전체 패키지 목록 및 버전 제약
