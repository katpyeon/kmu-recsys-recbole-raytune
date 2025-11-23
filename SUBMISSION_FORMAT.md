# 제출 형식 변경 완료

## 올바른 제출 형식

### 파일 형식
- **컬럼**: `user_id`, `item_ids`
- **구분자**: 아이템은 **공백(' ')으로 구분**
- **형태**: 사용자당 1행

### 예시
```csv
user_id,item_ids
U16127,P00244 P01918 P02258 P01790 P00801
U28891,P00356 P00278 P00450 P00596 P02247
U01934,P00453 P00783 P00006 P01809 P00068
```

## 변경 사항

### 모델 학습 파일 (5개)
1. `01-ease-raytune.py`
2. `02-lightgcn-raytune.py`
3. `02-lightgcn-raytune-opt.py`
4. `03-multivae-raytune.py`
5. `04-recvae-raytune.py`

**변경 내용:**
- 데이터 경로: `dataset/apply_train.csv` → `dataset/comp_train.csv`
- 입력 컬럼: `resume_seq`, `recruitment_seq` → `user_id`, `item_id`
- 출력 형식: 각 사용자당 1행, 아이템은 공백으로 구분

### 앙상블 파일 (2개)
1. `05-ensemble-rrf.py`
2. `06-ensemble-auto.py`

**변경 내용:**
- 개별 모델 제출 파일 읽기: 공백으로 구분된 item_ids를 split()으로 파싱
- 출력 형식: 각 사용자당 1행, 아이템은 공백으로 구분

## 주요 코드 변경

### Before (틀린 형식)
```python
# 여러 행 (틀림)
for user_id, recs in all_recommendations.items():
    for item_id in recs:
        result.append((user_id, item_id))  # ❌

submission = pd.DataFrame(result, columns=['user_id', 'item_ids'])
```

### After (올바른 형식)
```python
# 사용자당 1행 (정답)
for user_id, recs in all_recommendations.items():
    items_str = ' '.join(recs)  # ✅ 공백으로 구분
    result.append((user_id, items_str))

submission = pd.DataFrame(result, columns=['user_id', 'item_ids'])
```

## 참고: sample 디렉토리 노트북
제출 형식은 `sample/W12-comp_02_model_tuning.ipynb`에서 확인:
```python
top_k = pd.DataFrame([
    {DEFAULT_USER_COL: user_id, 'item_ids': ' '.join(items)}
    for user_id, items in result_topk.items()
])
```
