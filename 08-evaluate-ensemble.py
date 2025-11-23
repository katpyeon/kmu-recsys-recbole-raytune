#!/usr/bin/env python3
"""
앙상블 제출 파일 평가 스크립트

RecBole 테스트셋으로 앙상블의 실제 Recall@5, NDCG@5 계산
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_seed

print("=" * 80)
print("앙상블 제출 파일 평가")
print("=" * 80)
print()

# 데이터 준비
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

# 데이터셋 생성
config = Config(model='BPR', config_dict=config_dict)
init_seed(config['seed'], config['reproducibility'])
dataset = create_dataset(config)
train_data, valid_data, test_data = data_preparation(config, dataset)

print(f"✅ 데이터셋 준비 완료")
print(f"   Train: {len(train_data.dataset):,}개")
print(f"   Valid: {len(valid_data.dataset):,}개")
print(f"   Test: {len(test_data.dataset):,}개")
print()

# 앙상블 제출 파일 로드
ensemble_file = 'outputs/2025-11-23/submit_Ensemble_BORDA_20251123131619.csv'
print(f"앙상블 파일 로드: {ensemble_file}")

ensemble_df = pd.read_csv(ensemble_file)
print(f"✅ {len(ensemble_df):,}개 사용자 추천 로드")
print()

# 추천 결과를 딕셔너리로 변환 (user_id -> [item1, item2, ...])
recommendations = {}
for _, row in ensemble_df.iterrows():
    user_external = row['user_id']
    items_external = row['item_ids'].split()
    recommendations[user_external] = items_external

# 테스트셋 정답 추출
test_interactions = {}
for uid in test_data.dataset.inter_feat['user_id'].unique():
    user_external = dataset.id2token('user_id', uid.item())

    # 해당 사용자의 테스트 데이터 인덱스
    user_test_indices = (test_data.dataset.inter_feat['user_id'] == uid).nonzero(as_tuple=True)[0]

    # 테스트 아이템 (정답)
    test_items_internal = test_data.dataset.inter_feat['item_id'][user_test_indices]
    test_items_external = [dataset.id2token('item_id', iid.item()) for iid in test_items_internal]

    test_interactions[user_external] = set(test_items_external)

print(f"✅ 테스트셋 정답 추출: {len(test_interactions):,}명")
print()

# Recall@5, NDCG@5 계산
recall_at_5 = []
ndcg_at_5 = []

for user_id, true_items in test_interactions.items():
    if user_id not in recommendations:
        # 추천이 없는 경우 0점
        recall_at_5.append(0.0)
        ndcg_at_5.append(0.0)
        continue

    pred_items = recommendations[user_id][:5]  # Top 5

    # Recall@5
    hits = len(set(pred_items) & true_items)
    recall = hits / min(len(true_items), 5) if len(true_items) > 0 else 0.0
    recall_at_5.append(recall)

    # NDCG@5
    dcg = 0.0
    idcg = sum([1.0 / np.log2(i + 2) for i in range(min(len(true_items), 5))])

    for i, item in enumerate(pred_items):
        if item in true_items:
            dcg += 1.0 / np.log2(i + 2)

    ndcg = dcg / idcg if idcg > 0 else 0.0
    ndcg_at_5.append(ndcg)

# 평균 계산
avg_recall = np.mean(recall_at_5)
avg_ndcg = np.mean(ndcg_at_5)

print("=" * 80)
print("평가 결과")
print("=" * 80)
print()
print(f"📊 앙상블 성능 (테스트셋):")
print(f"   Recall@5: {avg_recall:.4f}")
print(f"   NDCG@5:   {avg_ndcg:.4f}")
print()

# 개별 모델과 비교
print("📊 개별 모델 성능 (검증셋, 참고용):")
print(f"   MultiVAE:  Recall@5 = 0.0870")
print(f"   RecVAE:    Recall@5 = 0.0868")
print(f"   LightGCN:  Recall@5 = 0.0849")
print(f"   EASE:      Recall@5 = 0.0718")
print()

# 개선도 계산
best_individual = 0.0870  # MultiVAE
improvement = avg_recall - best_individual

print(f"🎯 앙상블 개선도:")
if improvement > 0:
    print(f"   +{improvement:.4f} ({improvement/best_individual*100:+.1f}%) ✅ 개선!")
elif improvement == 0:
    print(f"   {improvement:+.4f} (동일)")
else:
    print(f"   {improvement:.4f} ({improvement/best_individual*100:.1f}%) ❌ 저하")
print()

print("=" * 80)
print("✅ 평가 완료")
print("=" * 80)
