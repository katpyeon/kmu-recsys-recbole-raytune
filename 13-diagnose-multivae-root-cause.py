#!/usr/bin/env python3
"""
MultiVAE Root Cause Analysis

3가지 가설 검증:
1. 데이터 분할 문제 (Test set이 representative하지 않음?)
2. MultiVAE 심각한 Overfitting (10개 아이템만 추천)
3. 평가 방식 차이 (Metric 계산 방법 차이?)
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_seed

print("=" * 80)
print("MultiVAE Root Cause Analysis")
print("=" * 80)
print()

# ============================================================
# 데이터 준비
# ============================================================
train_file = 'dataset/comp_train.csv'
df = pd.read_csv(train_file)
df.columns = [col.replace('\ufeff', '') for col in df.columns]

df_recbole = pd.DataFrame({
    'user_id:token': df['user_id'],
    'item_id:token': df['item_id'],
    'rating:float': 1.0
})

dataset_dir = 'dataset/kaggle_recsys'
os.makedirs(dataset_dir, exist_ok=True)
inter_file = os.path.join(dataset_dir, 'kaggle_recsys.inter')
df_recbole.to_csv(inter_file, sep='\t', index=False)

# RecBole 설정
DATASET_PATH = str(Path(__file__).parent / 'dataset')
config_dict = {
    'data_path': DATASET_PATH,
    'dataset': 'kaggle_recsys',
    'USER_ID_FIELD': 'user_id',
    'ITEM_ID_FIELD': 'item_id',
    'RATING_FIELD': 'rating',
    'load_col': {'inter': ['user_id', 'item_id', 'rating']},
    'eval_args': {
        'split': {'RS': [0.8, 0.1, 0.1]},
        'order': 'RO',
        'mode': 'full',
        'group_by': 'user'
    },
    'seed': 2024,
}

config = Config(model='BPR', config_dict=config_dict)
init_seed(config['seed'], config['reproducibility'])
dataset = create_dataset(config)
train_data, valid_data, test_data = data_preparation(config, dataset)

print(f"✅ 데이터셋 준비")
print(f"   전체: {len(df):,}개 interactions")
print(f"   Train: {len(train_data.dataset):,}개")
print(f"   Valid: {len(valid_data.dataset):,}개")
print(f"   Test: {len(test_data.dataset):,}개")
print()

# ============================================================
# 가설 1: 데이터 분할 문제
# ============================================================
print("=" * 80)
print("가설 1: 데이터 분할 문제 분석")
print("=" * 80)
print()

# 전체 데이터 통계
total_users = df['user_id'].nunique()
total_items = df['item_id'].nunique()
total_interactions = len(df)

print(f"📊 전체 데이터 통계:")
print(f"   사용자: {total_users:,}명")
print(f"   아이템: {total_items:,}개")
print(f"   Interactions: {total_interactions:,}개")
print(f"   평균 interaction/user: {total_interactions/total_users:.2f}")
print()

# Train/Valid/Test 분할 통계
train_users = train_data.dataset.inter_feat['user_id'].unique()
valid_users = valid_data.dataset.inter_feat['user_id'].unique()
test_users = test_data.dataset.inter_feat['user_id'].unique()

train_items = train_data.dataset.inter_feat['item_id'].unique()
valid_items = valid_data.dataset.inter_feat['item_id'].unique()
test_items = test_data.dataset.inter_feat['item_id'].unique()

print(f"📊 분할별 통계:")
print(f"   Train: {len(train_users):,}명, {len(train_items):,}개 아이템")
print(f"   Valid: {len(valid_users):,}명, {len(valid_items):,}개 아이템")
print(f"   Test:  {len(test_users):,}명, {len(test_items):,}개 아이템")
print()

# Test set의 representative성 검증
# Test에만 있는 아이템 (Cold-start items)
test_items_external = set([dataset.id2token('item_id', iid.item()) for iid in test_items])
train_items_external = set([dataset.id2token('item_id', iid.item()) for iid in train_items])
cold_start_items = test_items_external - train_items_external

print(f"🔍 Test Set Representative성:")
print(f"   Test에만 있는 아이템 (cold-start): {len(cold_start_items):,}개 ({len(cold_start_items)/len(test_items_external)*100:.1f}%)")
print()

if len(cold_start_items) > 0:
    print(f"⚠️  경고: Test에 {len(cold_start_items)}개 cold-start 아이템 존재!")
    print(f"   이는 모델이 절대 추천할 수 없는 아이템입니다.")
    print()

# ============================================================
# 가설 2: MultiVAE Severe Overfitting 분석
# ============================================================
print("=" * 80)
print("가설 2: MultiVAE Severe Overfitting 분석")
print("=" * 80)
print()

# MultiVAE 제출 파일 분석
multivae_file = 'outputs/2025-11-22/submit_MultiVAE_RayTune_20251122142530.csv'
multivae_df = pd.read_csv(multivae_file)

# 전체 추천된 아이템 분석
all_recommended_items = []
for items_str in multivae_df['item_ids']:
    all_recommended_items.extend(items_str.split())

item_counter = Counter(all_recommended_items)
unique_items = len(item_counter)
total_recommendations = len(all_recommended_items)

print(f"📊 MultiVAE 추천 통계:")
print(f"   총 추천 수: {total_recommendations:,}개")
print(f"   고유 아이템 수: {unique_items:,}개")
print(f"   커버리지: {unique_items/total_items*100:.2f}% (전체 아이템 대비)")
print()

# Top 20 가장 많이 추천된 아이템
print(f"📊 가장 많이 추천된 Top 20 아이템:")
top_20 = item_counter.most_common(20)
top_20_count = sum(count for _, count in top_20)
for i, (item_id, count) in enumerate(top_20, 1):
    pct = count / total_recommendations * 100
    print(f"   {i:2d}. {item_id}: {count:,}회 ({pct:.1f}%)")

print()
print(f"🔍 Top 20 아이템 집중도:")
print(f"   Top 20이 전체 추천의 {top_20_count/total_recommendations*100:.1f}% 차지")
print()

# 사용자당 추천 다양성
user_diversity = []
for items_str in multivae_df['item_ids']:
    items = items_str.split()
    user_diversity.append(len(set(items)))

print(f"📊 사용자당 추천 다양성:")
print(f"   평균: {np.mean(user_diversity):.2f}개 고유 아이템 (최대 5개)")
print(f"   중앙값: {np.median(user_diversity):.0f}개")
print(f"   최소: {np.min(user_diversity)}개, 최대: {np.max(user_diversity)}개")
print()

# 전체 데이터의 아이템 빈도와 비교
train_item_freq = df.groupby('item_id').size().sort_values(ascending=False)
top_20_popular = train_item_freq.head(20).index.tolist()

print(f"🔍 MultiVAE 추천 vs 전체 데이터 인기도:")
multivae_top_items = [item_id for item_id, _ in item_counter.most_common(20)]
overlap = len(set(multivae_top_items) & set(top_20_popular))
print(f"   MultiVAE Top 20과 실제 Top 20 겹침: {overlap}/20개")
print(f"   → MultiVAE는 {'인기 아이템만' if overlap >= 15 else '다양한 아이템을'} 추천")
print()

# ============================================================
# 가설 3: 평가 방식 차이
# ============================================================
print("=" * 80)
print("가설 3: 평가 방식 차이 분석")
print("=" * 80)
print()

# Test set 정답 추출
test_interactions = {}
for uid in test_data.dataset.inter_feat['user_id'].unique():
    user_external = dataset.id2token('user_id', uid.item())
    user_test_indices = (test_data.dataset.inter_feat['user_id'] == uid).nonzero(as_tuple=True)[0]
    test_items_internal = test_data.dataset.inter_feat['item_id'][user_test_indices]
    test_items_external = [dataset.id2token('item_id', iid.item()) for iid in test_items_internal]
    test_interactions[user_external] = set(test_items_external)

# MultiVAE 추천 파싱
recommendations = {}
for _, row in multivae_df.iterrows():
    user_external = row['user_id']
    items_external = row['item_ids'].split()
    recommendations[user_external] = items_external

# Recall@5, NDCG@5 계산 (우리 방식)
recall_at_5 = []
ndcg_at_5 = []
hit_count_dist = []

for user_id, true_items in test_interactions.items():
    if user_id not in recommendations:
        recall_at_5.append(0.0)
        ndcg_at_5.append(0.0)
        hit_count_dist.append(0)
        continue

    pred_items = recommendations[user_id][:5]

    # Recall@5
    hits = len(set(pred_items) & true_items)
    recall = hits / min(len(true_items), 5) if len(true_items) > 0 else 0.0
    recall_at_5.append(recall)
    hit_count_dist.append(hits)

    # NDCG@5
    dcg = 0.0
    idcg = sum([1.0 / np.log2(i + 2) for i in range(min(len(true_items), 5))])
    for i, item in enumerate(pred_items):
        if item in true_items:
            dcg += 1.0 / np.log2(i + 2)
    ndcg = dcg / idcg if idcg > 0 else 0.0
    ndcg_at_5.append(ndcg)

avg_recall = np.mean(recall_at_5)
avg_ndcg = np.mean(ndcg_at_5)

print(f"📊 우리 평가 방식 (Test Set):")
print(f"   Recall@5: {avg_recall:.4f}")
print(f"   NDCG@5: {avg_ndcg:.4f}")
print()

# Hit count 분포
hit_counter = Counter(hit_count_dist)
print(f"📊 Hit Count 분포 (Top 5 추천 중 몇 개가 정답?):")
for hits in sorted(hit_counter.keys()):
    count = hit_counter[hits]
    pct = count / len(test_interactions) * 100
    print(f"   {hits}개 히트: {count:,}명 ({pct:.1f}%)")
print()

# ============================================================
# 종합 분석
# ============================================================
print("=" * 80)
print("종합 분석")
print("=" * 80)
print()

print(f"🔍 가설 검증 결과:")
print()

print(f"1️⃣  데이터 분할 문제:")
if len(cold_start_items) > 10:
    print(f"   ⚠️  CRITICAL: Test에 {len(cold_start_items)}개 cold-start 아이템 존재")
    print(f"   → 이는 성능을 인위적으로 낮출 수 있음")
else:
    print(f"   ✅ Test set은 representative함 (cold-start 아이템 {len(cold_start_items)}개)")
print()

print(f"2️⃣  MultiVAE Severe Overfitting:")
if unique_items < 50:
    print(f"   ❌ CRITICAL: 단 {unique_items}개 아이템만 추천")
    print(f"   → 심각한 overfitting / popularity bias")
    print(f"   → 커버리지: {unique_items/total_items*100:.2f}% (목표: >10%)")
else:
    print(f"   ✅ 추천 다양성: {unique_items}개 아이템")
print()

print(f"3️⃣  평가 방식 차이:")
print(f"   우리 Test Recall@5: {avg_recall:.4f}")
print(f"   Public LB Recall@5: 0.197")
print(f"   격차: {abs(avg_recall - 0.197):.4f} (2.4배)")
print()
if abs(avg_recall - 0.197) > 0.05:
    print(f"   ❌ CRITICAL: 2.4배 격차는 비정상적")
    print(f"   → 평가 데이터가 다르거나 metric 계산이 다름")
    print(f"   → Public LB는 다른 test set을 사용하는 것으로 추정")
else:
    print(f"   ✅ 격차가 합리적 범위")
print()

# ============================================================
# 최종 진단
# ============================================================
print("=" * 80)
print("최종 진단 및 권장사항")
print("=" * 80)
print()

print(f"🎯 핵심 문제:")
print()

if unique_items < 50:
    print(f"1. MultiVAE 심각한 Overfitting (단 {unique_items}개 아이템만 추천)")
    print(f"   원인:")
    print(f"   - Dropout 너무 높음 (0.519)")
    print(f"   - Learning rate 너무 낮음 (5.34e-05)")
    print(f"   - Anneal cap 너무 높음 (0.4)")
    print()
    print(f"   권장 조치:")
    print(f"   - Dropout: 0.3~0.4로 낮추기")
    print(f"   - Learning rate: 1e-4 ~ 1e-3로 높이기")
    print(f"   - Anneal cap: 0.1~0.2로 낮추기")
    print(f"   - Regularization 완화하여 모델 capacity 확보")
    print()

if abs(avg_recall - 0.197) > 0.05:
    print(f"2. 평가 데이터 불일치 (Test={avg_recall:.4f} vs Public LB=0.197)")
    print(f"   원인:")
    print(f"   - Public LB는 다른 test set 사용")
    print(f"   - 우리는 80/10/10 split의 10% test만 사용")
    print(f"   - Public LB는 별도의 hidden test set 사용")
    print()
    print(f"   권장 조치:")
    print(f"   - 전체 데이터로 재학습 (No split)")
    print(f"   - Validation은 cross-validation 사용")
    print(f"   - Public LB 점수를 신뢰하고 최적화")
    print()

if len(cold_start_items) > 10:
    print(f"3. Cold-start 아이템 문제 ({len(cold_start_items)}개)")
    print(f"   권장 조치:")
    print(f"   - Data split 방식 변경 (item-based split 대신 time-based)")
    print()

print("=" * 80)
print("✅ 분석 완료")
print("=" * 80)
