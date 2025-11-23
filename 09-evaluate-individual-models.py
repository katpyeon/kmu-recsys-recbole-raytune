#!/usr/bin/env python3
"""
개별 모델 제출 파일 평가 스크립트 (RecVAE 제외)

RecBole 테스트셋으로 개별 모델의 실제 Recall@5, NDCG@5 계산
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_seed

print("=" * 80)
print("개별 모델 제출 파일 평가 (테스트셋)")
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

# 테스트셋 정답 추출
test_interactions = {}
for uid in test_data.dataset.inter_feat['user_id'].unique():
    user_external = dataset.id2token('user_id', uid.item())
    user_test_indices = (test_data.dataset.inter_feat['user_id'] == uid).nonzero(as_tuple=True)[0]
    test_items_internal = test_data.dataset.inter_feat['item_id'][user_test_indices]
    test_items_external = [dataset.id2token('item_id', iid.item()) for iid in test_items_internal]
    test_interactions[user_external] = set(test_items_external)

print(f"✅ 테스트셋 정답 추출: {len(test_interactions):,}명")
print()

# 개별 모델 파일 (RecVAE 제외)
model_files = {
    'MultiVAE': 'outputs/2025-11-22/submit_MultiVAE_RayTune_20251122142530.csv',
    'LightGCN': 'outputs/2025-11-22/submit_LightGCN_RayTune_20251122142021.csv',
    'EASE': 'outputs/2025-11-22/submit_EASE_RayTune_20251122141144.csv',
}

# 평가 함수
def evaluate_model(submission_file, model_name):
    """모델 제출 파일을 테스트셋으로 평가"""
    # 제출 파일 로드
    sub_df = pd.read_csv(submission_file)

    # 추천 결과를 딕셔너리로 변환
    recommendations = {}
    for _, row in sub_df.iterrows():
        user_external = row['user_id']
        items_external = row['item_ids'].split()
        recommendations[user_external] = items_external

    # Recall@5, NDCG@5 계산
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

    avg_recall = np.mean(recall_at_5)
    avg_ndcg = np.mean(ndcg_at_5)

    return avg_recall, avg_ndcg

# 모든 모델 평가
print("=" * 80)
print("개별 모델 평가 결과 (테스트셋)")
print("=" * 80)
print()

results = {}
for model_name, file_path in model_files.items():
    if os.path.exists(file_path):
        print(f"평가 중: {model_name}...")
        recall, ndcg = evaluate_model(file_path, model_name)
        results[model_name] = {'recall@5': recall, 'ndcg@5': ndcg}
        print(f"  ✅ {model_name:10s}: Recall@5 = {recall:.4f}, NDCG@5 = {ndcg:.4f}")
    else:
        print(f"  ⚠️  {model_name:10s}: 파일 없음")

print()

# 앙상블 평가
print("=" * 80)
print("앙상블 평가 (참고용)")
print("=" * 80)
print()

ensemble_file = 'outputs/2025-11-23/submit_Ensemble_BORDA_20251123131619.csv'
if os.path.exists(ensemble_file):
    recall, ndcg = evaluate_model(ensemble_file, 'Ensemble')
    results['Ensemble (Borda)'] = {'recall@5': recall, 'ndcg@5': ndcg}
    print(f"  Ensemble (Borda): Recall@5 = {recall:.4f}, NDCG@5 = {ndcg:.4f}")

print()

# 최종 비교
print("=" * 80)
print("최종 성능 비교 (테스트셋)")
print("=" * 80)
print()

# Recall@5 기준 정렬
sorted_results = sorted(results.items(), key=lambda x: x[1]['recall@5'], reverse=True)

print(f"{'순위':<5} {'모델':<20} {'Recall@5':<12} {'NDCG@5':<12}")
print("-" * 50)

for rank, (model_name, metrics) in enumerate(sorted_results, 1):
    recall = metrics['recall@5']
    ndcg = metrics['ndcg@5']
    marker = "🏆" if rank == 1 else "  "
    print(f"{marker} {rank:<3} {model_name:<20} {recall:.4f}      {ndcg:.4f}")

print()

# 최고 모델과의 차이
best_model = sorted_results[0][0]
best_recall = sorted_results[0][1]['recall@5']

print(f"🏆 최고 성능 모델: {best_model} (Recall@5 = {best_recall:.4f})")
print()

if 'Ensemble (Borda)' in results:
    ensemble_recall = results['Ensemble (Borda)']['recall@5']
    diff = ensemble_recall - best_recall
    pct = (diff / best_recall * 100) if best_recall > 0 else 0

    print(f"📊 앙상블 vs 최고 모델:")
    if diff > 0:
        print(f"   +{diff:.4f} ({pct:+.1f}%) ✅ 개선!")
    elif diff == 0:
        print(f"   {diff:+.4f} (동일)")
    else:
        print(f"   {diff:.4f} ({pct:.1f}%) ❌ 저하")

print()
print("=" * 80)
print("✅ 평가 완료")
print("=" * 80)
