#!/usr/bin/env python3
"""
2개 모델 앙상블 평가
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_seed

print("=" * 80)
print("2개 모델 앙상블 평가")
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

config = Config(model='BPR', config_dict=config_dict)
init_seed(config['seed'], config['reproducibility'])
dataset = create_dataset(config)
train_data, valid_data, test_data = data_preparation(config, dataset)

# 테스트셋 정답
test_interactions = {}
for uid in test_data.dataset.inter_feat['user_id'].unique():
    user_external = dataset.id2token('user_id', uid.item())
    user_test_indices = (test_data.dataset.inter_feat['user_id'] == uid).nonzero(as_tuple=True)[0]
    test_items_internal = test_data.dataset.inter_feat['item_id'][user_test_indices]
    test_items_external = [dataset.id2token('item_id', iid.item()) for iid in test_items_internal]
    test_interactions[user_external] = set(test_items_external)

print(f"✅ 테스트셋 준비: {len(test_interactions):,}명\n")

# 평가 함수
def evaluate_model(submission_file, model_name):
    sub_df = pd.read_csv(submission_file)
    recommendations = {}
    for _, row in sub_df.iterrows():
        user_external = row['user_id']
        items_external = row['item_ids'].split()
        recommendations[user_external] = items_external

    recall_at_5 = []
    ndcg_at_5 = []

    for user_id, true_items in test_interactions.items():
        if user_id not in recommendations:
            recall_at_5.append(0.0)
            ndcg_at_5.append(0.0)
            continue

        pred_items = recommendations[user_id][:5]

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

    return np.mean(recall_at_5), np.mean(ndcg_at_5)

# 평가
print("=" * 80)
print("평가 결과")
print("=" * 80)
print()

results = {}

# 개별 모델
models = {
    'MultiVAE': 'outputs/2025-11-22/submit_MultiVAE_RayTune_20251122142530.csv',
    'LightGCN': 'outputs/2025-11-22/submit_LightGCN_RayTune_20251122142021.csv',
}

for model_name, file_path in models.items():
    recall, ndcg = evaluate_model(file_path, model_name)
    results[model_name] = {'recall@5': recall, 'ndcg@5': ndcg}
    print(f"{model_name:15s}: Recall@5 = {recall:.4f}, NDCG@5 = {ndcg:.4f}")

# 2개 모델 앙상블
ensemble_file = 'outputs/2025-11-23/submit_Ensemble_2Models_20251123135535.csv'
recall, ndcg = evaluate_model(ensemble_file, '2-Model Ensemble')
results['2-Model Ensemble'] = {'recall@5': recall, 'ndcg@5': ndcg}
print(f"{'2-Model Ensemble':15s}: Recall@5 = {recall:.4f}, NDCG@5 = {ndcg:.4f}")

# 4개 모델 앙상블 (비교)
ensemble_4_file = 'outputs/2025-11-23/submit_Ensemble_BORDA_20251123131619.csv'
if os.path.exists(ensemble_4_file):
    recall, ndcg = evaluate_model(ensemble_4_file, '4-Model Ensemble')
    results['4-Model Ensemble'] = {'recall@5': recall, 'ndcg@5': ndcg}
    print(f"{'4-Model Ensemble':15s}: Recall@5 = {recall:.4f}, NDCG@5 = {ndcg:.4f}")

print()

# 최종 비교
print("=" * 80)
print("최종 성능 순위")
print("=" * 80)
print()

sorted_results = sorted(results.items(), key=lambda x: x[1]['recall@5'], reverse=True)

print(f"{'순위':<5} {'모델':<20} {'Recall@5':<12} {'NDCG@5':<12} {'차이'}")
print("-" * 60)

best_recall = sorted_results[0][1]['recall@5']

for rank, (model_name, metrics) in enumerate(sorted_results, 1):
    recall = metrics['recall@5']
    ndcg = metrics['ndcg@5']
    diff = recall - best_recall
    diff_pct = (diff / best_recall * 100) if best_recall > 0 else 0
    marker = "🏆" if rank == 1 else "  "

    if diff == 0:
        diff_str = "-"
    elif diff > 0:
        diff_str = f"+{diff:.4f} (+{diff_pct:.1f}%)"
    else:
        diff_str = f"{diff:.4f} ({diff_pct:.1f}%)"

    print(f"{marker} {rank:<3} {model_name:<20} {recall:.4f}      {ndcg:.4f}      {diff_str}")

print()
print("=" * 80)
print("✅ 평가 완료")
print("=" * 80)
