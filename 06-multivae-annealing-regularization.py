#!/usr/bin/env python3
"""
RecBole AutoML - MultiVAE Annealing + Regularization Exploration

제안 1: 한 번도 튜닝하지 않은 핵심 VAE 파라미터 탐색
========================================================

배경:
- 지금까지 learning_rate, dropout, anneal_cap, latent_dim, MLP는 모두 탐색함
- 하지만 VAE 학습의 핵심 파라미터들은 한 번도 튜닝 안 함:
  * total_anneal_steps: KL annealing 속도 (기본값 200,000)
  * weight_decay: L2 regularization (기본값 0.0)

가설:
- 단순한 데이터셋 → 더 빠른 annealing (50k-100k)이 효과적일 수 있음
- 희소성 99.9% → weight_decay로 regularization 필요할 수 있음

탐색 공간:
- total_anneal_steps: [50k, 100k, 200k, 400k]
- weight_decay: [1e-6, 1e-3] (log uniform)
- 나머지: 검증된 최적값 근처로 고정

예상 성능: 5-10% 개선 가능 (근본 파라미터 튜닝)
"""

import os
import sys
import warnings
import pandas as pd
import numpy as np
import time
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
print("MultiVAE Annealing + Regularization Exploration")
print("=" * 80)
print("✅ 라이브러리 로드 완료\n")

# ============================================================
# 1. 디바이스
# ============================================================
print("=" * 80)
print("1. 디바이스 선택")
print("=" * 80)

if torch.cuda.is_available():
    device = 'cuda'
    print(f"🚀 디바이스: CUDA ({torch.cuda.get_device_name(0)})")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = 'mps'
    print("🍎 디바이스: MPS (Apple Silicon)")
else:
    device = 'cpu'
    print("💻 디바이스: CPU")

print(f"PyTorch: {torch.__version__}\n")

# ============================================================
# 2. Ray 초기화
# ============================================================
print("=" * 80)
print("2. Ray 초기화")
print("=" * 80)

import ray

if ray.is_initialized():
    ray.shutdown()

total_cpus = os.cpu_count() or 4

if device == 'cuda':
    num_cpus = total_cpus
    num_gpus = 1
    print(f"🎮 CUDA: CPU {num_cpus}코어, GPU 1개")
elif device == 'mps':
    num_cpus = total_cpus
    num_gpus = 0
    print(f"🍎 MPS: CPU {num_cpus}코어")
else:
    num_cpus = total_cpus
    num_gpus = 0
    print(f"💻 CPU: {num_cpus}코어")

ray.init(
    ignore_reinit_error=True,
    include_dashboard=False,
    num_cpus=num_cpus,
    num_gpus=num_gpus,
    _temp_dir=None,
    _metrics_export_port=None,
    configure_logging=False,
)

print("✅ Ray 초기화 완료\n")

# ============================================================
# 3. 데이터 로딩
# ============================================================
print("=" * 80)
print("3. 데이터 로딩")
print("=" * 80)

train_file = 'dataset/comp_train.csv'
df = pd.read_csv(train_file)
df.columns = [col.replace('\ufeff', '') for col in df.columns]

print(f"✅ 데이터: {len(df):,}개 상호작용")
print(f"   사용자: {df['user_id'].nunique():,}, 아이템: {df['item_id'].nunique():,}")
print(f"   희소성: {1 - len(df) / (df['user_id'].nunique() * df['item_id'].nunique()):.4%}")

df_recbole = pd.DataFrame({
    'user_id:token': df['user_id'],
    'item_id:token': df['item_id'],
    'rating:float': 1.0
})

dataset_dir = 'dataset/kaggle_recsys'
os.makedirs(dataset_dir, exist_ok=True)
inter_file = os.path.join(dataset_dir, 'kaggle_recsys.inter')
df_recbole.to_csv(inter_file, sep='\t', index=False)

print(f"✅ RecBole 데이터셋: {inter_file}\n")

# ============================================================
# 4. Ray Tune 설정
# ============================================================
print("=" * 80)
print("4. Ray Tune 설정")
print("=" * 80)

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

# Annealing + Regularization 탐색 공간
# 기존 최적값: latent_dim=128, mlp=[600], dropout~0.3-0.55, anneal_cap=0.2, lr~0.0003-0.0006
search_space = {
    # Lock to known optimal values
    'latent_dimension': tune.choice([128]),
    'mlp_hidden_size': tune.choice([[600]]),
    'dropout_prob': tune.uniform(0.25, 0.35),
    'anneal_cap': tune.choice([0.2]),
    'learning_rate': tune.loguniform(3e-4, 8e-4),

    # NEW: Never tuned before!
    'total_anneal_steps': tune.choice([50000, 100000, 200000, 400000]),
    'weight_decay': tune.loguniform(1e-6, 1e-3),
}

print("✅ 설정 완료")
print(f"   모델: {MODEL_NAME} - Annealing + Regularization")
print(f"   타겟: Recall@5")
print(f"\n🔍 탐색 공간:")
print(f"   latent_dimension: 128 (고정)")
print(f"   mlp_hidden_size: [600] (고정)")
print(f"   dropout_prob: [0.25, 0.35]")
print(f"   anneal_cap: 0.2 (고정)")
print(f"   learning_rate: [3e-4, 8e-4]")
print(f"\n   🆕 total_anneal_steps: [50k, 100k, 200k, 400k]")
print(f"   🆕 weight_decay: [1e-6, 1e-3]")
print(f"\n💡 예상: 5-10% 개선 (VAE 핵심 파라미터)\n")

# ============================================================
# 5. Trainable 함수
# ============================================================
print("=" * 80)
print("5. Trainable 함수 정의")
print("=" * 80)

def train_recbole(config_params):
    """Ray Tune trainable - Annealing + Regularization experiment"""
    from ray import train

    config_dict = base_config.copy()
    config_dict.update({
        'model': MODEL_NAME,
        'latent_dimension': int(config_params['latent_dimension']),
        'mlp_hidden_size': config_params['mlp_hidden_size'],
        'dropout_prob': config_params['dropout_prob'],
        'anneal_cap': config_params['anneal_cap'],
        'learning_rate': config_params['learning_rate'],
        'total_anneal_steps': int(config_params['total_anneal_steps']),
        'weight_decay': config_params['weight_decay'],
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
        train.report({
            'recall@5': 0.0,
            'ndcg@5': 0.0,
            'recall@10': 0.0,
        })

print("✅ Trainable 함수 정의 완료\n")

# ============================================================
# 6. Ray Tune 실행
# ============================================================
print("=" * 80)
print("6. Ray Tune 최적화")
print("=" * 80)

start_time = time.time()

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

ray_results_path = str(Path('./ray_results').resolve())

if device == 'cuda':
    resources_per_trial = {"cpu": 1, "gpu": 0.16}
    max_concurrent_trials = 6
    print(f"🎮 CUDA: {max_concurrent_trials}개 동시 실행")
elif device == 'mps':
    resources_per_trial = {"cpu": 2}
    max_concurrent_trials = None
    print(f"🍎 MPS: 자동 병렬화")
else:
    resources_per_trial = {"cpu": 2}
    max_concurrent_trials = None
    print(f"💻 CPU: 자동 병렬화")

tuner = tune.Tuner(
    tune.with_resources(train_recbole, resources=resources_per_trial),
    param_space=search_space,
    tune_config=tune.TuneConfig(
        scheduler=scheduler,
        search_alg=search_alg,
        num_samples=30,
        max_concurrent_trials=max_concurrent_trials,
    ),
    run_config=RunConfig(
        name='recbole_multivae_anneal_reg',
        storage_path=ray_results_path,
    ),
)

print("\n🚀 최적화 시작 (30 trials)...\n")
results = tuner.fit()

print(f"\n✅ 최적화 완료 (소요: {time.time() - start_time:.2f}초)\n")

# ============================================================
# 7. 결과 분석
# ============================================================
print("=" * 80)
print("7. 결과 분석")
print("=" * 80)

best_result = results.get_best_result(metric='recall@5', mode='max')
best_config = best_result.config
best_metrics = best_result.metrics

print("\n🏆 최적 하이퍼파라미터:")
print(f"   latent_dimension: {int(best_config['latent_dimension'])}")
print(f"   mlp_hidden_size: {best_config['mlp_hidden_size']}")
print(f"   dropout_prob: {best_config['dropout_prob']:.4f}")
print(f"   anneal_cap: {best_config['anneal_cap']:.4f}")
print(f"   learning_rate: {best_config['learning_rate']:.6f}")
print(f"   🆕 total_anneal_steps: {int(best_config['total_anneal_steps']):,}")
print(f"   🆕 weight_decay: {best_config['weight_decay']:.6f}")

print(f"\n🎯 검증 성능:")
print(f"   Recall@5: {best_metrics['recall@5']:.4f}")
print(f"   NDCG@5: {best_metrics['ndcg@5']:.4f}")

# 파라미터별 성능 분석
results_df = results.get_dataframe()
successful_df = results_df[results_df['recall@5'] > 0]

print(f"\n📊 total_anneal_steps별 평균 성능:")
anneal_perf = successful_df.groupby('config/total_anneal_steps')['recall@5'].agg(['mean', 'count'])
anneal_perf = anneal_perf.sort_values('mean', ascending=False)
for steps, row in anneal_perf.iterrows():
    print(f"   {int(steps):>7,} steps: {row['mean']:.4f} (n={int(row['count'])})")

print(f"\n📊 weight_decay 분포:")
print(f"   최소: {successful_df['config/weight_decay'].min():.6f}")
print(f"   중앙: {successful_df['config/weight_decay'].median():.6f}")
print(f"   최대: {successful_df['config/weight_decay'].max():.6f}")

# Baseline 비교
BASELINE_RECALL = 0.087
improvement = (best_metrics['recall@5'] - BASELINE_RECALL) / BASELINE_RECALL * 100

print(f"\n📈 Baseline 대비:")
print(f"   Baseline: {BASELINE_RECALL:.4f}")
print(f"   최고점: {best_metrics['recall@5']:.4f}")
print(f"   개선율: {improvement:+.2f}%")

# 최적 파라미터 저장
import json
t_params = pd.Timestamp.now()
params_output_dir = f"outputs/{t_params.year}-{t_params.month:02d}-{t_params.day:02d}"
os.makedirs(params_output_dir, exist_ok=True)
params_filename = f"{params_output_dir}/best_hyperparams_multivae_anneal_reg_{t_params.year}{t_params.month:02d}{t_params.day:02d}{t_params.hour:02d}{t_params.minute:02d}{t_params.second:02d}.json"

best_params_to_save = {
    'hyperparameters': {
        'latent_dimension': int(best_config['latent_dimension']),
        'mlp_hidden_size': best_config['mlp_hidden_size'],
        'dropout_prob': float(best_config['dropout_prob']),
        'anneal_cap': float(best_config['anneal_cap']),
        'learning_rate': float(best_config['learning_rate']),
        'total_anneal_steps': int(best_config['total_anneal_steps']),
        'weight_decay': float(best_config['weight_decay']),
    },
    'validation_metrics': {
        'recall@5': float(best_metrics['recall@5']),
        'ndcg@5': float(best_metrics['ndcg@5']),
        'recall@10': float(best_metrics['recall@10'])
    },
    'improvement_vs_baseline': {
        'baseline_recall@5': BASELINE_RECALL,
        'improvement_pct': float(improvement),
    },
    'anneal_steps_analysis': {
        str(int(k)): {'mean': float(v['mean']), 'count': int(v['count'])}
        for k, v in anneal_perf.iterrows()
    },
    'timestamp': t_params.strftime('%Y-%m-%d %H:%M:%S'),
    'experiment': 'Annealing + Regularization',
}

with open(params_filename, 'w') as f:
    json.dump(best_params_to_save, f, indent=2)

print(f"\n💾 결과 저장: {params_filename}\n")

# ============================================================
# 8. 최적 모델 최종 학습
# ============================================================
print("=" * 80)
print("8. 최적 모델 최종 학습")
print("=" * 80)

start_time = time.time()

final_config_dict = base_config.copy()
final_config_dict.update({
    'model': MODEL_NAME,
    'latent_dimension': int(best_config['latent_dimension']),
    'mlp_hidden_size': best_config['mlp_hidden_size'],
    'dropout_prob': best_config['dropout_prob'],
    'anneal_cap': best_config['anneal_cap'],
    'learning_rate': best_config['learning_rate'],
    'total_anneal_steps': int(best_config['total_anneal_steps']),
    'weight_decay': best_config['weight_decay'],
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

print(f"\n✅ 최종 학습 완료 (소요: {time.time() - start_time:.2f}초)")
print(f"\n📊 검증 성능:")
print(f"   Recall@5:  {best_valid_result['recall@5']:.4f}")
print(f"   NDCG@5:    {best_valid_result['ndcg@5']:.4f}")
print(f"\n📊 테스트 성능:")
print(f"   Recall@5:  {test_result['recall@5']:.4f}")
print(f"   NDCG@5:    {test_result['ndcg@5']:.4f}\n")

# ============================================================
# 9. 추천 생성 및 제출 파일
# ============================================================
print("=" * 80)
print("9. 추천 생성")
print("=" * 80)

start_time = time.time()

all_users = dataset.inter_feat['user_id'].unique()
all_recommendations = {}
topk = 5

model.eval()
with torch.no_grad():
    for i, user_id in enumerate(all_users):
        user_external = dataset.id2token('user_id', user_id.item())

        topk_scores, topk_indices = full_sort_topk(
            [user_id.item()],
            model,
            test_data,
            k=topk,
            device=config['device']
        )

        topk_items_internal = topk_indices[0].cpu().tolist()
        items_external = [dataset.id2token('item_id', int(item)) for item in topk_items_internal]
        all_recommendations[user_external] = items_external

        if (i + 1) % 1000 == 0:
            print(f"   {i + 1}/{len(all_users)} 사용자 처리")

print(f"\n✅ 추천 생성 완료 (소요: {time.time() - start_time:.2f}초)\n")

# ============================================================
# 10. 제출 파일 생성
# ============================================================
print("=" * 80)
print("10. 제출 파일 생성")
print("=" * 80)

result = []
for user_id, recs in all_recommendations.items():
    items_str = ' '.join(recs)
    result.append((user_id, items_str))

submission = pd.DataFrame(result, columns=['user_id', 'item_ids'])

t = pd.Timestamp.now()
output_dir = f"outputs/{t.year}-{t.month:02d}-{t.day:02d}"
os.makedirs(output_dir, exist_ok=True)
filename = f"{output_dir}/submit_{MODEL_NAME}_Anneal_Reg_{t.year}{t.month:02d}{t.day:02d}{t.hour:02d}{t.minute:02d}{t.second:02d}.csv"

submission.to_csv(filename, index=False)

print(f"\n✅ 제출 파일 생성 완료!")
print(f"   파일: {filename}")
print(f"   검증 Recall@5: {best_valid_score:.4f}")
print(f"   테스트 Recall@5: {test_result['recall@5']:.4f}")

# ============================================================
# 11. 최종 요약
# ============================================================
print("\n" + "=" * 80)
print("MultiVAE Annealing + Regularization - 결과 요약")
print("=" * 80)

print(f"\n🤖 실험 정보:")
print(f"   총 trials: {len(results_df)}")
print(f"   성공 trials: {len(successful_df)}")

print(f"\n🏆 최고 성능:")
print(f"   검증 Recall@5: {best_valid_result['recall@5']:.4f}")
print(f"   테스트 Recall@5: {test_result['recall@5']:.4f}")
print(f"   total_anneal_steps: {int(best_config['total_anneal_steps']):,}")
print(f"   weight_decay: {best_config['weight_decay']:.6f}")

print(f"\n📈 Baseline 대비:")
print(f"   Baseline: {BASELINE_RECALL:.4f}")
print(f"   개선율: {improvement:+.2f}%")

if improvement > 0:
    print(f"\n✨ 성공! Annealing + Regularization으로 개선")
else:
    print(f"\n⚠️  개선 없음. Baseline이 여전히 최고")

print(f"\n💾 출력 파일:")
print(f"   제출: {filename}")
print(f"   분석: {params_filename}")

print("\n" + "=" * 80)
print("✅ 모든 작업 완료!")
print("=" * 80)

ray.shutdown()
