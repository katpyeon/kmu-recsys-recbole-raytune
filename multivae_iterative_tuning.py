#!/usr/bin/env python3
"""
RecBole MultiVAE - Iterative Hyperparameter Tuning with Automatic Stopping

전략:
- 짧은 실험 반복 (12→10→8 trials)
- 성능 추적 및 자동 중단
- 최고 성능 모델로 제출 파일 생성

중단 조건:
- 2회 연속 성능 하락
- 3회 연속 0.3% 미만 개선
- 최대 5회 반복
"""

import os
import sys
import warnings
import pandas as pd
import numpy as np
import time
import json
from datetime import datetime
from pathlib import Path
import torch

# RecBole imports
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_seed, get_model, get_trainer
from recbole.utils.case_study import full_sort_topk

# Ray Tune imports
from ray import tune
from ray.train import RunConfig
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch

# Ray Tune 버그 우회
import ray.tune.experimental.output as tune_output
import ray.train._internal.storage as train_storage

original_get_air_verbosity = tune_output.get_air_verbosity
def patched_get_air_verbosity(verbose):
    if isinstance(verbose, str):
        return 1
    return original_get_air_verbosity(verbose)
tune_output.get_air_verbosity = patched_get_air_verbosity

original_storage_init = train_storage.StorageContext.__init__
def patched_storage_init(self, *args, **kwargs):
    if 'sync_config' in kwargs and isinstance(kwargs['sync_config'], str):
        from ray.train import SyncConfig
        kwargs['sync_config'] = SyncConfig()
    return original_storage_init(self, *args, **kwargs)
train_storage.StorageContext.__init__ = patched_storage_init

original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)
torch.load = patched_torch_load

warnings.filterwarnings('ignore')

print("=" * 80)
print("MultiVAE 반복 튜닝 - 자동 중단 및 최적화")
print("=" * 80)
print()

# ============================================================
# 디바이스 선택
# ============================================================
if torch.cuda.is_available():
    device = 'cuda'
    print(f"🚀 디바이스: CUDA ({torch.cuda.get_device_name(0)})")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = 'mps'
    print("🍎 디바이스: MPS")
else:
    device = 'cpu'
    print("💻 디바이스: CPU")

# ============================================================
# Ray 초기화
# ============================================================
import ray

if ray.is_initialized():
    ray.shutdown()

total_cpus = os.cpu_count() or 4

if device == 'cuda':
    num_cpus = total_cpus
    num_gpus = 1
elif device == 'mps':
    num_cpus = total_cpus
    num_gpus = 0
else:
    num_cpus = total_cpus
    num_gpus = 0

ray.init(
    ignore_reinit_error=True,
    include_dashboard=False,
    num_cpus=num_cpus,
    num_gpus=num_gpus,
    _temp_dir=None,
    _metrics_export_port=None,
    configure_logging=False,
)

print(f"✅ Ray 초기화 완료\n")

# ============================================================
# 데이터 로딩
# ============================================================
print("=" * 80)
print("데이터 로딩")
print("=" * 80)

train_file = 'dataset/comp_train.csv'
df = pd.read_csv(train_file)
df.columns = [col.replace('\ufeff', '') for col in df.columns]

print(f"✅ 데이터 로드: {len(df):,}개 상호작용")
print(f"   사용자: {df['user_id'].nunique():,}, 아이템: {df['item_id'].nunique():,}")
print(f"   희소성: {1 - len(df) / (df['user_id'].nunique() * df['item_id'].nunique()):.4%}\n")

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

# ============================================================
# 기본 설정
# ============================================================
MODEL_NAME = 'MultiVAE'
DATASET_PATH = str(Path(__file__).parent / 'dataset')

if device == 'cuda':
    train_batch_size = 4096
    eval_batch_size = 102400
elif device == 'mps':
    train_batch_size = 2048
    eval_batch_size = 4096
else:
    train_batch_size = 2048
    eval_batch_size = 4096

base_config = {
    'data_path': DATASET_PATH,
    'dataset': 'kaggle_recsys',
    'USER_ID_FIELD': 'user_id',
    'ITEM_ID_FIELD': 'item_id',
    'RATING_FIELD': 'rating',
    'load_col': {'inter': ['user_id', 'item_id', 'rating']},
    'train_neg_sample_args': None,
    'eval_args': {
        'split': {'RS': [0.8, 0.1, 0.1]},
        'order': 'RO',
        'mode': 'full',
        'group_by': 'user'
    },
    'metrics': ['Recall', 'NDCG', 'MRR'],
    'topk': [5, 10, 20],
    'valid_metric': 'Recall@5',
    'device': device,
    'epochs': 100,
    'stopping_step': 10,
    'train_batch_size': train_batch_size,
    'eval_batch_size': eval_batch_size,
    'seed': 2024,
    'reproducibility': False,
    'show_progress': False,
    'worker': 4,
}

# ============================================================
# Trainable 함수
# ============================================================
def train_recbole(config_params):
    from ray import train

    config_dict = base_config.copy()
    config_dict.update({
        'model': MODEL_NAME,
        'latent_dimension': int(config_params['latent_dimension']),
        'mlp_hidden_size': config_params['mlp_hidden_size'],
        'dropout_prob': config_params['dropout_prob'],
        'anneal_cap': config_params['anneal_cap'],
        'learning_rate': config_params['learning_rate'],
    })

    try:
        config = Config(model=MODEL_NAME, config_dict=config_dict)
        init_seed(config['seed'], config['reproducibility'])

        dataset = create_dataset(config)
        train_data, valid_data, test_data = data_preparation(config, dataset)

        model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
        trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

        best_valid_score, best_valid_result = trainer.fit(
            train_data, valid_data, verbose=False, show_progress=False
        )

        train.report({
            'recall@5': best_valid_result['recall@5'],
            'ndcg@5': best_valid_result['ndcg@5'],
            'recall@10': best_valid_result['recall@10'],
        })
    except Exception as e:
        print(f"❌ Trial 실패: {str(e)}")
        train.report({'recall@5': 0.0, 'ndcg@5': 0.0, 'recall@10': 0.0})

# ============================================================
# 반복 탐색 공간 정의 함수
# ============================================================
def get_search_space(iteration, iterations):
    """반복 번호와 이전 결과를 기반으로 탐색 공간 반환"""

    if iteration == 1:
        # Iteration 1: 최적값 주변 세밀 탐색
        return {
            'latent_dimension': tune.choice([128, 200]),
            'mlp_hidden_size': tune.choice([[600], [512]]),
            'dropout_prob': tune.uniform(0.28, 0.65),
            'anneal_cap': tune.choice([0.1, 0.2, 0.3]),
            'learning_rate': tune.loguniform(3e-4, 1.5e-3),
        }

    # 이전 최고 결과 분석
    prev_best = max(iterations, key=lambda x: x['recall@5'])
    prev_config = prev_best['config']
    prev_recall = prev_best['recall@5']

    # Baseline과 비교
    baseline_recall = 0.087
    improvement = prev_recall - baseline_recall

    if improvement > 0.002:  # 0.2% 이상 개선
        # 개선됨 → 축소하여 정밀 탐색
        best_lr = prev_config['learning_rate']
        best_dropout = prev_config['dropout_prob']

        return {
            'latent_dimension': tune.choice([int(prev_config['latent_dimension'])]),
            'mlp_hidden_size': tune.choice([prev_config['mlp_hidden_size']]),
            'dropout_prob': tune.uniform(max(0.25, best_dropout - 0.1), min(0.7, best_dropout + 0.1)),
            'anneal_cap': tune.choice([prev_config['anneal_cap']]),
            'learning_rate': tune.loguniform(best_lr * 0.7, best_lr * 1.3),
        }
    else:
        # 정체 → 다른 방향 탐색
        return {
            'latent_dimension': tune.choice([200, 256]),
            'mlp_hidden_size': tune.choice([[600]]),
            'dropout_prob': tune.uniform(0.2, 0.75),
            'anneal_cap': tune.choice([0.2, 0.4, 0.5]),
            'learning_rate': tune.loguniform(3e-4, 2e-3),
        }

# ============================================================
# 중단 조건 체크
# ============================================================
def should_stop(iterations):
    """중단 조건 체크"""
    if len(iterations) < 2:
        return False

    recalls = [it['recall@5'] for it in iterations]

    # 조건 1: 2회 연속 하락
    if len(recalls) >= 2:
        if recalls[-1] < recalls[-2] and recalls[-2] < recalls[-3] if len(recalls) >= 3 else False:
            return True

    # 조건 2: 3회 연속 0.3% 미만 개선
    if len(recalls) >= 3:
        improvements = [recalls[i] - recalls[i-1] for i in range(-2, 0)]
        if all(imp < 0.003 for imp in improvements):
            return True

    # 조건 3: 최대 반복 횟수
    if len(iterations) >= 5:
        return True

    return False

def get_stop_reason(iterations):
    """중단 이유 반환"""
    recalls = [it['recall@5'] for it in iterations]

    if len(iterations) >= 5:
        return "최대 반복 횟수 도달 (5회)"

    if len(recalls) >= 2:
        if recalls[-1] < recalls[-2]:
            if len(recalls) >= 3 and recalls[-2] < recalls[-3]:
                return "2회 연속 성능 하락"

    if len(recalls) >= 3:
        improvements = [recalls[i] - recalls[i-1] for i in range(-2, 0)]
        if all(imp < 0.003 for imp in improvements):
            return "3회 연속 0.3% 미만 개선 (정체)"

    return "알 수 없음"

# ============================================================
# 반복 튜닝 실행
# ============================================================
print("=" * 80)
print("반복 튜닝 시작")
print("=" * 80)
print()

iterations = []
baseline_recall = 0.087

ray_results_path = str(Path('./ray_results').resolve())

if device == 'cuda':
    resources_per_trial = {"cpu": 1, "gpu": 0.16}
    max_concurrent_trials = 6
elif device == 'mps':
    resources_per_trial = {"cpu": 2}
    max_concurrent_trials = None
else:
    resources_per_trial = {"cpu": 2}
    max_concurrent_trials = None

for iteration in range(1, 6):
    print("=" * 80)
    print(f"ITERATION {iteration}")
    print("=" * 80)

    # 탐색 공간 결정
    search_space = get_search_space(iteration, iterations)

    # Trial 수 결정
    if iteration == 1:
        num_trials = 12
    elif iteration == 2:
        num_trials = 10
    else:
        num_trials = 8

    print(f"\n🔍 탐색 공간:")
    for key, value in search_space.items():
        print(f"   {key}: {value}")
    print(f"\n📊 Trial 수: {num_trials}\n")

    # Ray Tune 실행
    scheduler = ASHAScheduler(
        metric='recall@5',
        mode='max',
        max_t=100,
        grace_period=10,
        reduction_factor=2,
    )

    search_alg = OptunaSearch(
        metric='recall@5',
        mode='max',
    )

    tuner = tune.Tuner(
        tune.with_resources(train_recbole, resources=resources_per_trial),
        param_space=search_space,
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            search_alg=search_alg,
            num_samples=num_trials,
            max_concurrent_trials=max_concurrent_trials,
        ),
        run_config=RunConfig(
            name=f'recbole_multivae_iter{iteration}',
            storage_path=ray_results_path,
        ),
    )

    start_time = time.time()
    results = tuner.fit()
    elapsed = time.time() - start_time

    # 최고 결과 추출
    best_result = results.get_best_result(metric='recall@5', mode='max')
    best_config = best_result.config
    best_metrics = best_result.metrics

    recall = best_metrics['recall@5']
    improvement = recall - baseline_recall

    # 결과 저장
    iterations.append({
        'iteration': iteration,
        'recall@5': recall,
        'ndcg@5': best_metrics['ndcg@5'],
        'config': best_config,
        'improvement': improvement,
        'elapsed': elapsed,
        'num_trials': num_trials
    })

    # 결과 출력
    print(f"\n{'=' * 80}")
    print(f"ITERATION {iteration} 완료")
    print(f"{'=' * 80}")
    print(f"🎯 Recall@5: {recall:.4f} ({improvement:+.4f} vs baseline)")
    print(f"⏱️  소요 시간: {elapsed:.1f}초")
    print(f"\n최적 하이퍼파라미터:")
    print(f"   learning_rate: {best_config['learning_rate']:.6f}")
    print(f"   dropout_prob: {best_config['dropout_prob']:.4f}")
    print(f"   anneal_cap: {best_config['anneal_cap']:.4f}")
    print(f"   latent_dimension: {int(best_config['latent_dimension'])}")
    print(f"   mlp_hidden_size: {best_config['mlp_hidden_size']}\n")

    # 중단 조건 체크
    if should_stop(iterations):
        print(f"🛑 중단: {get_stop_reason(iterations)}\n")
        break

# ============================================================
# 최종 결과 및 비교표
# ============================================================
print("=" * 80)
print("반복 튜닝 최종 결과")
print("=" * 80)
print()

# 최고 성능 찾기
best_iteration = max(iterations, key=lambda x: x['recall@5'])

print(f"✅ 최고 성능: Iteration {best_iteration['iteration']}")
print(f"   Recall@5: {best_iteration['recall@5']:.4f}")
print(f"   Baseline 대비: {best_iteration['improvement']:+.4f} ({best_iteration['improvement']/baseline_recall*100:+.1f}%)\n")

# 비교표
print("=" * 80)
print("성능 비교표")
print("=" * 80)
print()
print(f"{'Version':<15} {'Recall@5':<10} {'개선':<10} {'LR':<12} {'Dropout':<10} {'Anneal':<8} {'Trials':<8}")
print("-" * 80)
print(f"{'Baseline':<15} {baseline_recall:<10.4f} {'-':<10} {'0.000517':<12} {'0.302':<10} {'0.2':<8} {'30':<8}")

for it in iterations:
    cfg = it['config']
    print(f"{'Iter ' + str(it['iteration']):<15} {it['recall@5']:<10.4f} "
          f"{it['improvement']:+.4f}    {cfg['learning_rate']:<12.6f} "
          f"{cfg['dropout_prob']:<10.4f} {cfg['anneal_cap']:<8.2f} {it['num_trials']:<8}")

print()

# ============================================================
# 최고 성능 모델로 최종 학습 및 제출 파일 생성
# ============================================================
print("=" * 80)
print("최고 성능 모델로 최종 학습 및 제출 파일 생성")
print("=" * 80)
print()

best_config = best_iteration['config']

final_config_dict = base_config.copy()
final_config_dict.update({
    'model': MODEL_NAME,
    'latent_dimension': int(best_config['latent_dimension']),
    'mlp_hidden_size': best_config['mlp_hidden_size'],
    'dropout_prob': best_config['dropout_prob'],
    'anneal_cap': best_config['anneal_cap'],
    'learning_rate': best_config['learning_rate'],
    'show_progress': True,
})

config = Config(model=MODEL_NAME, config_dict=final_config_dict)
init_seed(config['seed'], config['reproducibility'])

dataset = create_dataset(config)
train_data, valid_data, test_data = data_preparation(config, dataset)

model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)
test_result = trainer.evaluate(test_data)

print(f"\n✅ 최종 학습 완료")
print(f"   검증 Recall@5: {best_valid_result['recall@5']:.4f}")
print(f"   테스트 Recall@5: {test_result['recall@5']:.4f}\n")

# 추천 생성
print("추천 생성 중...")
all_users = dataset.inter_feat['user_id'].unique()
all_recommendations = {}

model.eval()
with torch.no_grad():
    for i, user_id in enumerate(all_users):
        user_external = dataset.id2token('user_id', user_id.item())

        topk_scores, topk_indices = full_sort_topk(
            [user_id.item()],
            model,
            test_data,
            k=5,
            device=config['device']
        )

        topk_items_internal = topk_indices[0].cpu().tolist()
        items_external = [dataset.id2token('item_id', int(item)) for item in topk_items_internal]
        all_recommendations[user_external] = items_external

# 제출 파일 생성
result = []
for user_id, recs in all_recommendations.items():
    items_str = ' '.join(recs)
    result.append((user_id, items_str))

submission = pd.DataFrame(result, columns=['user_id', 'item_ids'])

t = pd.Timestamp.now()
output_dir = f"outputs/{t.year}-{t.month:02d}-{t.day:02d}"
os.makedirs(output_dir, exist_ok=True)
filename = f"{output_dir}/submit_MultiVAE_Iterative_Best_{t.year}{t.month:02d}{t.day:02d}{t.hour:02d}{t.minute:02d}{t.second:02d}.csv"

submission.to_csv(filename, index=False)

print(f"\n✅ 제출 파일 생성 완료!")
print(f"   파일명: {filename}")
print(f"   Recall@5: {test_result['recall@5']:.4f}")

# 파라미터 저장
params_filename = f"{output_dir}/best_hyperparams_multivae_iterative_{t.year}{t.month:02d}{t.day:02d}{t.hour:02d}{t.minute:02d}{t.second:02d}.json"

params_to_save = {
    'best_iteration': best_iteration['iteration'],
    'hyperparameters': {
        'latent_dimension': int(best_config['latent_dimension']),
        'mlp_hidden_size': best_config['mlp_hidden_size'],
        'dropout_prob': float(best_config['dropout_prob']),
        'anneal_cap': float(best_config['anneal_cap']),
        'learning_rate': float(best_config['learning_rate'])
    },
    'validation_metrics': {
        'recall@5': float(best_valid_result['recall@5']),
        'ndcg@5': float(best_valid_result['ndcg@5']),
    },
    'test_metrics': {
        'recall@5': float(test_result['recall@5']),
        'ndcg@5': float(test_result['ndcg@5']),
    },
    'iterations_summary': [
        {
            'iteration': it['iteration'],
            'recall@5': float(it['recall@5']),
            'improvement': float(it['improvement']),
        }
        for it in iterations
    ],
    'timestamp': t.strftime('%Y-%m-%d %H:%M:%S'),
}

with open(params_filename, 'w') as f:
    json.dump(params_to_save, f, indent=2)

print(f"   파라미터: {params_filename}")

print("\n" + "=" * 80)
print("✅ 모든 작업 완료!")
print("=" * 80)

ray.shutdown()
