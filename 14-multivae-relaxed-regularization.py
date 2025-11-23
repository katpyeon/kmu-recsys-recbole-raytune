#!/usr/bin/env python3
"""
MultiVAE Regularization 완화 실험

문제: 현재 MultiVAE는 단 10개 아이템만 추천 (심각한 overfitting)
원인: 과도한 regularization
- Dropout: 0.519 (너무 높음)
- Learning rate: 5.34e-05 (너무 낮음)
- Anneal cap: 0.4 (너무 높음)

해결: Regularization 완화 + 전체 데이터 활용
- Dropout: 0.2~0.4
- Learning rate: 1e-4~5e-3
- Anneal cap: 0.1~0.2
- Data split: 90/10/0 (전체 데이터 최대 활용)

목표: 추천 다양성 10개 → 100개+, Public LB 0.197 → 0.20+
"""

import os
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_seed

from ray import tune, train, air
from ray.air import session
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler

print("=" * 80)
print("MultiVAE Regularization 완화 실험")
print("=" * 80)
print()

# ============================================================
# 1. 데이터 준비
# ============================================================
print("=" * 80)
print("1. 데이터 준비")
print("=" * 80)
print()

train_file = 'dataset/comp_train.csv'
df = pd.read_csv(train_file)
df.columns = [col.replace('\ufeff', '') for col in df.columns]

print(f"원본 데이터: {len(df):,}개 interactions")
print(f"사용자: {df['user_id'].nunique():,}명")
print(f"아이템: {df['item_id'].nunique():,}개")
print()

# RecBole 형식 변환
df_recbole = pd.DataFrame({
    'user_id:token': df['user_id'],
    'item_id:token': df['item_id'],
    'rating:float': 1.0
})

dataset_dir = 'dataset/kaggle_recsys'
os.makedirs(dataset_dir, exist_ok=True)
inter_file = os.path.join(dataset_dir, 'kaggle_recsys.inter')
df_recbole.to_csv(inter_file, sep='\t', index=False)

print(f"✅ RecBole 데이터셋 생성: {inter_file}")
print()

# ============================================================
# 2. Ray Tune Training Function
# ============================================================

DATASET_PATH = str(Path(__file__).parent / 'dataset')

def train_multivae_with_config(config_hyperparams):
    """Ray Tune training function"""

    # Base config (90/10/0 split - 전체 데이터 최대 활용)
    base_config = {
        'data_path': DATASET_PATH,
        'dataset': 'kaggle_recsys',
        'USER_ID_FIELD': 'user_id',
        'ITEM_ID_FIELD': 'item_id',
        'RATING_FIELD': 'rating',
        'load_col': {'inter': ['user_id', 'item_id', 'rating']},
        'eval_args': {
            'split': {'RS': [0.9, 0.1, 0.0]},  # 90% train, 10% validation, 0% test
            'order': 'RO',
            'mode': 'full',
            'group_by': 'user'
        },
        'metrics': ['Recall', 'NDCG', 'MRR'],
        'topk': [5, 10, 20],
        'valid_metric': 'Recall@5',
        'epochs': 100,
        'stopping_step': 10,
        'train_batch_size': 4096,
        'eval_batch_size': 102400,
        'seed': 2024,
        'reproducibility': True,
    }

    # Merge with hyperparameters
    full_config = {**base_config, **config_hyperparams}

    # RecBole config
    config = Config(model='MultiVAE', config_dict=full_config)
    init_seed(config['seed'], config['reproducibility'])

    # Dataset
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # Model
    from recbole.model.general_recommender import MultiVAE
    model = MultiVAE(config, train_data.dataset).to(config['device'])

    # Trainer
    from recbole.trainer import Trainer
    trainer = Trainer(config, model)

    # Train
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=False, show_progress=False
    )

    # Report to Ray Tune
    train.report({
        'recall@5': best_valid_result['recall@5'],
        'ndcg@5': best_valid_result['ndcg@5'],
        'mrr@5': best_valid_result['mrr@5'],
    })

# ============================================================
# 3. Ray Tune 설정
# ============================================================
print("=" * 80)
print("2. Ray Tune Hyperparameter Search 설정")
print("=" * 80)
print()

# 개선된 Search Space (Regularization 완화)
search_space = {
    # 핵심 변경: Regularization 완화
    'dropout_prob': tune.uniform(0.2, 0.4),         # 0.25~0.65 → 0.2~0.4
    'learning_rate': tune.loguniform(1e-4, 5e-3),   # 5e-5~5e-3 → 1e-4~5e-3
    'anneal_cap': tune.choice([0.1, 0.2]),          # 0.1~0.4 → 0.1~0.2

    # 기존 유지
    'latent_dimension': tune.choice([128, 200, 256]),
    'mlp_hidden_size': tune.choice([[600], [512]]),
}

print("📊 Search Space (Regularization 완화):")
print(f"   dropout_prob: 0.2~0.4 (이전: 0.25~0.65)")
print(f"   learning_rate: 1e-4~5e-3 (이전: 5e-5~5e-3)")
print(f"   anneal_cap: [0.1, 0.2] (이전: [0.1, 0.2, 0.3, 0.4])")
print(f"   latent_dimension: [128, 200, 256]")
print(f"   mlp_hidden_size: [[600], [512]]")
print()

print("🎯 목표:")
print(f"   추천 다양성: 10개 → 100개+")
print(f"   Validation Recall@5: 0.0865 → 0.10+")
print(f"   Public LB: 0.197 → 0.20+")
print()

# Optuna TPE Sampler
optuna_search = OptunaSearch(
    metric='recall@5',
    mode='max'
)

# ASHA Scheduler (metric/mode will be set in TuneConfig)
asha_scheduler = ASHAScheduler(
    max_t=100,
    grace_period=10,
    reduction_factor=2
)

# ============================================================
# 4. Ray Tune 실행
# ============================================================
print("=" * 80)
print("3. Ray Tune 실행 시작")
print("=" * 80)
print()

# Ray init with /tmp/ray directory
import ray
ray.init(_temp_dir='/tmp/ray', ignore_reinit_error=True)

num_samples = 30
print(f"Trials: {num_samples}")
print(f"예상 시간: 1-2시간")
print()

tuner = tune.Tuner(
    train_multivae_with_config,
    param_space=search_space,
    tune_config=tune.TuneConfig(
        search_alg=optuna_search,
        scheduler=asha_scheduler,
        num_samples=num_samples,
        metric='recall@5',
        mode='max',
    ),
    run_config=air.RunConfig(
        name='multivae_relaxed_regularization',
        stop={'training_iteration': 1},
        verbose=1,
    )
)

results = tuner.fit()

# ============================================================
# 5. Best Result 분석
# ============================================================
print()
print("=" * 80)
print("4. Best Result 분석")
print("=" * 80)
print()

best_result = results.get_best_result(metric='recall@5', mode='max')
best_config = best_result.config
best_metrics = best_result.metrics

print(f"🏆 Best Hyperparameters:")
print(f"   dropout_prob: {best_config['dropout_prob']:.4f}")
print(f"   learning_rate: {best_config['learning_rate']:.6f}")
print(f"   anneal_cap: {best_config['anneal_cap']}")
print(f"   latent_dimension: {best_config['latent_dimension']}")
print(f"   mlp_hidden_size: {best_config['mlp_hidden_size']}")
print()

print(f"🏆 Best Validation Metrics:")
print(f"   Recall@5: {best_metrics['recall@5']:.4f}")
print(f"   NDCG@5: {best_metrics['ndcg@5']:.4f}")
print(f"   MRR@5: {best_metrics['mrr@5']:.4f}")
print()

# 이전 결과와 비교
print(f"📊 성능 비교:")
print(f"   이전 (v3): Recall@5 = 0.0865, NDCG@5 = 0.0576")
print(f"   현재 (relaxed): Recall@5 = {best_metrics['recall@5']:.4f}, NDCG@5 = {best_metrics['ndcg@5']:.4f}")
improvement = (best_metrics['recall@5'] - 0.0865) / 0.0865 * 100
print(f"   개선도: {improvement:+.1f}%")
print()

# ============================================================
# 6. Best Model 재학습 및 제출 파일 생성
# ============================================================
print("=" * 80)
print("5. Best Model 재학습 및 제출 파일 생성")
print("=" * 80)
print()

# Best config로 재학습
final_config_dict = {
    'data_path': DATASET_PATH,
    'dataset': 'kaggle_recsys',
    'USER_ID_FIELD': 'user_id',
    'ITEM_ID_FIELD': 'item_id',
    'RATING_FIELD': 'rating',
    'load_col': {'inter': ['user_id', 'item_id', 'rating']},
    'eval_args': {
        'split': {'RS': [0.9, 0.1, 0.0]},
        'order': 'RO',
        'mode': 'full',
        'group_by': 'user'
    },
    'metrics': ['Recall', 'NDCG', 'MRR'],
    'topk': [5, 10, 20],
    'valid_metric': 'Recall@5',
    'epochs': 100,
    'stopping_step': 10,
    'train_batch_size': 4096,
    'eval_batch_size': 102400,
    'seed': 2024,
    'reproducibility': True,

    # Best hyperparameters
    'dropout_prob': best_config['dropout_prob'],
    'learning_rate': best_config['learning_rate'],
    'anneal_cap': best_config['anneal_cap'],
    'latent_dimension': best_config['latent_dimension'],
    'mlp_hidden_size': best_config['mlp_hidden_size'],
}

config = Config(model='MultiVAE', config_dict=final_config_dict)
init_seed(config['seed'], config['reproducibility'])

dataset = create_dataset(config)
train_data, valid_data, test_data = data_preparation(config, dataset)

print(f"✅ 데이터 준비 완료")
print(f"   Train: {len(train_data.dataset):,}개")
print(f"   Valid: {len(valid_data.dataset):,}개")
print()

# Model
from recbole.model.general_recommender import MultiVAE
model = MultiVAE(config, train_data.dataset).to(config['device'])

# Trainer
from recbole.trainer import Trainer
trainer = Trainer(config, model)

# Train
print("학습 시작...")
best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, saved=True, show_progress=True)

print()
print(f"✅ 학습 완료")
print(f"   Best Validation Recall@5: {best_valid_result['recall@5']:.4f}")
print()

# ============================================================
# 7. 제출 파일 생성
# ============================================================
print("=" * 80)
print("6. 제출 파일 생성")
print("=" * 80)
print()

# 전체 사용자에 대한 추천 생성
all_users = dataset.inter_feat['user_id'].unique()
print(f"전체 사용자 수: {len(all_users):,}명")

recommendations = []

for user_internal in all_users:
    user_external = dataset.id2token('user_id', user_internal.item())

    # User tensor
    user_tensor = torch.tensor([user_internal.item()]).to(config['device'])

    # Predict scores
    with torch.no_grad():
        scores = model.full_sort_predict(user_tensor)

    # Top 5
    _, top_indices = torch.topk(scores, k=5)
    top_items_internal = top_indices.cpu().numpy()[0]

    # Internal → External
    top_items_external = [dataset.id2token('item_id', int(iid)) for iid in top_items_internal]

    recommendations.append({
        'user_id': user_external,
        'item_ids': ' '.join(top_items_external)
    })

submission_df = pd.DataFrame(recommendations)

# 통계
all_items = set()
for items_str in submission_df['item_ids']:
    all_items.update(items_str.split())

print(f"✅ 추천 생성 완료")
print(f"   총 사용자: {len(submission_df):,}명")
print(f"   고유 아이템: {len(all_items):,}개")
print()

print(f"📊 추천 다양성 비교:")
print(f"   이전 (v3): 10개 아이템 (0.31% 커버리지)")
print(f"   현재 (relaxed): {len(all_items):,}개 아이템 ({len(all_items)/df['item_id'].nunique()*100:.2f}% 커버리지)")
print()

if len(all_items) > 50:
    print(f"✅ 성공! 추천 다양성이 크게 개선되었습니다!")
else:
    print(f"⚠️  경고: 여전히 추천 다양성이 낮습니다.")
print()

# 저장
t = pd.Timestamp.now()
output_dir = f"outputs/{t.year}-{t.month:02d}-{t.day:02d}"
os.makedirs(output_dir, exist_ok=True)
filename = f"{output_dir}/submit_MultiVAE_Relaxed_{t.year}{t.month:02d}{t.day:02d}{t.hour:02d}{t.minute:02d}{t.second:02d}.csv"

submission_df.to_csv(filename, index=False)

print(f"✅ 제출 파일 저장: {filename}")
print()

# ============================================================
# 8. Best Hyperparameters 저장
# ============================================================
import json

params_file = f"{output_dir}/best_hyperparams_multivae_relaxed_{t.year}{t.month:02d}{t.day:02d}{t.hour:02d}{t.minute:02d}{t.second:02d}.json"

params_data = {
    'hyperparameters': {
        'latent_dimension': int(best_config['latent_dimension']),
        'mlp_hidden_size': best_config['mlp_hidden_size'],
        'dropout_prob': float(best_config['dropout_prob']),
        'anneal_cap': float(best_config['anneal_cap']),
        'learning_rate': float(best_config['learning_rate']),
    },
    'validation_metrics': {
        'recall@5': float(best_metrics['recall@5']),
        'ndcg@5': float(best_metrics['ndcg@5']),
        'mrr@5': float(best_metrics['mrr@5']),
    },
    'diversity_metrics': {
        'unique_items': len(all_items),
        'coverage_pct': float(len(all_items) / df['item_id'].nunique() * 100),
    },
    'improvements': {
        'recall_improvement_pct': float(improvement),
        'diversity_improvement': f"{len(all_items)}개 (이전: 10개)",
    }
}

with open(params_file, 'w') as f:
    json.dump(params_data, f, indent=2)

print(f"✅ Hyperparameters 저장: {params_file}")
print()

# ============================================================
# 최종 요약
# ============================================================
print("=" * 80)
print("최종 요약")
print("=" * 80)
print()

print(f"🎯 실험 결과:")
print(f"   Validation Recall@5: {best_metrics['recall@5']:.4f} (이전: 0.0865)")
print(f"   추천 다양성: {len(all_items):,}개 (이전: 10개)")
print(f"   개선도: {improvement:+.1f}%")
print()

print(f"💾 출력 파일:")
print(f"   제출 파일: {filename}")
print(f"   Hyperparameters: {params_file}")
print()

print(f"📊 다음 단계:")
print(f"   1. 제출 파일을 Kaggle에 제출하여 Public LB 확인")
print(f"   2. Public LB > 0.20이면 성공!")
print(f"   3. 실험 B (100% 데이터 재학습) 진행")
print()

print("=" * 80)
print("✅ 모든 작업 완료!")
print("=" * 80)
