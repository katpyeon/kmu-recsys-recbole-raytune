# MultiVAE 개선 전략

## 문제 진단 요약

### 발견된 문제

1. **MultiVAE Severe Overfitting** (CRITICAL)
   - 단 10개 아이템만 추천 (0.31% 커버리지)
   - Top 6 아이템이 98% 차지
   - 현재 하이퍼파라미터가 모델 capacity를 심각하게 제한

2. **평가 데이터 불일치** (CRITICAL)
   - Test Recall@5 = 0.0819
   - Public LB = 0.197 (2.4배 차이)
   - Public LB는 별도의 hidden test set 사용
   - 우리의 80/10/10 split test는 Public LB와 무관

3. **Cold-start 문제** (WARNING)
   - Test에 64개 아이템이 Train에 없음
   - Random split의 한계

---

## 개선 전략 (우선순위별)

### 🎯 전략 1: 전체 데이터로 재학습 (최우선)

**이유:**
- Public LB는 별도의 test set 사용
- 우리의 10% test split은 성능 측정에 무의미
- 전체 데이터로 학습해야 Public LB 최적화 가능

**실행 방안:**
```python
config_dict = {
    'eval_args': {
        'split': {'RS': [0.9, 0.1, 0.0]},  # 90% train, 10% validation, 0% test
        'order': 'RO',
        'mode': 'full',
        'group_by': 'user'
    },
}
```

**또는 (더 공격적):**
```python
config_dict = {
    'eval_args': {
        'split': {'RS': [1.0, 0.0, 0.0]},  # 100% train, validation 없음
        'order': 'RO',
        'mode': 'full',
        'group_by': 'user'
    },
    'epochs': 50,  # Early stopping 없이 고정 epoch
}
```

**장점:**
- Public LB에 직접 최적화
- 더 많은 데이터로 학습
- Cold-start 문제 해결

**단점:**
- Overfitting 위험 증가
- Validation 없이 hyperparameter 선택 어려움

**권장:**
- 90/10 split으로 시작
- Best hyperparameter로 100% 재학습

---

### 🎯 전략 2: MultiVAE Regularization 완화 (매우 중요)

**현재 문제:**
- Dropout 0.519 → 모델 capacity 절반 이상 파괴
- Learning rate 5.34e-05 → 너무 느린 학습
- Anneal cap 0.4 → 과도한 KL regularization

**개선된 하이퍼파라미터 범위:**

```python
search_space = {
    # 핵심 변경사항
    'dropout_prob': tune.uniform(0.2, 0.4),      # 0.25~0.65 → 0.2~0.4
    'learning_rate': tune.loguniform(1e-4, 5e-3), # 5e-5~5e-3 → 1e-4~5e-3
    'anneal_cap': tune.choice([0.1, 0.2]),        # 0.1~0.4 → 0.1~0.2

    # 기존 유지
    'latent_dimension': tune.choice([128, 200, 256]),
    'mlp_hidden_size': tune.choice([[600], [512]]),
}
```

**이유:**
- Dropout 낮추기 → 모델 capacity 확보 → 다양한 아이템 학습
- Learning rate 높이기 → 빠른 수렴 → 더 나은 local optima
- Anneal cap 낮추기 → KL penalty 완화 → 더 rich한 latent representation

**기대 효과:**
- 추천 아이템 다양성 10개 → 100개+
- Recall@5 향상 (더 많은 사용자에게 relevant 추천)

---

### 🎯 전략 3: 1위팀 전략 모방

**1위팀 분석:**
- MultiVAE, LightGCN, EASE 모두 **비슷한 성능** (균형)
- 3개 앙상블로 0.201 달성
- 우리는 MultiVAE만 0.197, 나머지는 저조

**우리의 문제:**
- MultiVAE: Public LB 0.197 (좋음)
- LightGCN: Public LB 불명 (추정 0.15~0.18)
- EASE: Public LB 불명 (추정 0.10~0.15)
- **불균형** → 앙상블 효과 없음

**개선 방안:**

1. **LightGCN 재튜닝:**
   - 현재 Test Recall@5 = 0.0777 (MultiVAE 0.0819와 5% 차이)
   - 목표: Public LB 0.19+ (MultiVAE와 동등)

2. **EASE 재튜닝:**
   - 현재 Test Recall@5 = 0.0657 (매우 저조)
   - 목표: Public LB 0.18+

3. **균형잡힌 3개 모델 확보 후 앙상블:**
   - 예상: 0.19 + 0.19 + 0.18 → 앙상블 0.20+

---

## 실험 계획

### Phase 1: MultiVAE 긴급 수정 (1-2시간)

**실험 A: Regularization 완화**
```python
# 14-multivae-relaxed-regularization.py
search_space = {
    'dropout_prob': tune.uniform(0.2, 0.4),
    'learning_rate': tune.loguniform(1e-4, 5e-3),
    'anneal_cap': tune.choice([0.1, 0.2]),
    'latent_dimension': tune.choice([128, 200, 256]),
    'mlp_hidden_size': tune.choice([[600], [512]]),
}

# Data split: 90/10/0 (전체 데이터 최대 활용)
config_dict = {
    'eval_args': {
        'split': {'RS': [0.9, 0.1, 0.0]},
    },
}

# Ray Tune 설정
num_samples = 30
```

**예상 결과:**
- 추천 아이템 다양성: 10개 → 100개+
- Test Recall@5: 0.08 → 0.10+
- Public LB: 0.197 → 0.20+

---

**실험 B: 100% 데이터 재학습**
```python
# 15-multivae-full-data-retrain.py
# 실험 A의 best hyperparameter 사용
# 100% 데이터로 고정 50 epochs 학습
```

**예상 결과:**
- Public LB: 0.20 → 0.21+

---

### Phase 2: LightGCN 개선 (2-3시간)

**실험 C: LightGCN 집중 튜닝**
```python
# 16-lightgcn-balanced-tuning.py
# 목표: MultiVAE와 동등한 성능 (Public LB 0.19+)

search_space = {
    'embedding_size': tune.choice([64, 128, 256]),
    'n_layers': tune.choice([2, 3, 4]),
    'learning_rate': tune.loguniform(1e-4, 5e-3),
    'reg_weight': tune.loguniform(1e-5, 1e-2),
}

# 90/10 split
num_samples = 30
```

---

### Phase 3: EASE 개선 (1-2시간)

**실험 D: EASE 재튜닝**
```python
# 17-ease-improved-tuning.py
# EASE는 hyperparameter 적음 (reg_weight만)

search_space = {
    'reg_weight': tune.loguniform(1, 1000),
}

num_samples = 20
```

---

### Phase 4: 최종 앙상블 (30분)

**3개 균형잡힌 모델 앙상블:**
```python
# 18-final-balanced-ensemble.py
weights = {
    'MultiVAE': 0.35,   # 0.20+
    'LightGCN': 0.35,   # 0.19+
    'EASE': 0.30,       # 0.18+
}

# Borda Count
```

**목표:**
- Public LB: 0.20 → 0.21+ (1위 근접)

---

## 시간 예상

- **Phase 1 (MultiVAE):** 2시간
- **Phase 2 (LightGCN):** 3시간
- **Phase 3 (EASE):** 2시간
- **Phase 4 (앙상블):** 30분

**총 소요 시간: 7-8시간**

---

## 우선순위

1. **긴급:** 실험 A (MultiVAE regularization 완화)
   - 단 10개 아이템 문제 해결
   - 가장 빠른 성능 개선 기대

2. **중요:** 실험 B (100% 데이터 재학습)
   - Public LB 직접 최적화

3. **중요:** 실험 C (LightGCN 튜닝)
   - 앙상블을 위한 균형

4. **선택:** 실험 D + Phase 4
   - 0.004 격차 극복을 위한 최종 시도

---

## 예상 성능 개선 경로

```
현재:
- MultiVAE: 0.197 (Public LB)
- 앙상블: 불가 (불균형)

Phase 1 완료:
- MultiVAE: 0.20+ (Public LB)

Phase 1-2 완료:
- MultiVAE: 0.20+
- LightGCN: 0.19+
- 2-model 앙상블: 0.205+

Phase 1-4 완료:
- 3-model 앙상블: 0.21+
- 1위 (0.201) 근접 또는 초과
```

---

## 핵심 인사이트

1. **Public LB ≠ 우리 Test set**
   - 80/10/10 split test는 의미 없음
   - 전체 데이터 학습이 필수

2. **MultiVAE overfitting의 역설**
   - Test Recall@5는 낮지만 (0.0819)
   - Public LB는 높음 (0.197)
   - 단 10개 아이템 추천이 Public에서는 효과적?
   - → **아니다, regularization 완화로 개선 가능**

3. **1위와의 격차는 앙상블 균형**
   - 0.004 (0.197 → 0.201)는 운이 아님
   - 3개 균형잡힌 모델의 효과
   - 우리도 균형 확보하면 도달 가능
