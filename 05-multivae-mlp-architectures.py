#!/usr/bin/env python3
"""
RecBole AutoML - MultiVAE Multi-layer MLP Architecture Exploration

제안 4: Multi-layer MLP 아키텍처 탐색
배경:
- 지금까지 single-layer MLP만 탐색: [[600], [512]]
- Multi-layer MLP는 한 번도 시도하지 않음
- 계층적 패턴 학습 가능성 검증

탐색 방향:
1. Single-layer: [[600], [512], [400], [800]]
2. Two-layer (감소): [[600,300], [512,256], [400,200]]
3. Two-layer (큰 구조): [[800,400]]
4. Three-layer: [[600,400,200]]

기타 하이퍼파라미터:
- latent_dimension: [128, 200] (검증된 범위)
- dropout_prob: [0.25, 0.35] (최적값 0.302 근처)
- anneal_cap: 0.2 (최적값 고정)
- learning_rate: [3e-4, 8e-4] (최적값 0.000517 근처)

예상 성능: 낮은 개선(1-3%) - 논문에서 single-layer 권장
하지만 이 특정 데이터셋에서는 다를 수 있음
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

# Ray Tune 버그 우회: Monkey Patch
import ray.tune.experimental.output as tune_output
import ray.train._internal.storage as train_storage

# verbose 타입 버그 수정
original_get_air_verbosity = tune_output.get_air_verbosity
def patched_get_air_verbosity(verbose):
    if isinstance(verbose, str):
        return 1
    return original_get_air_verbosity(verbose)
tune_output.get_air_verbosity = patched_get_air_verbosity

# StorageContext __init__ 타입 버그 수정
original_storage_init = train_storage.StorageContext.__init__
def patched_storage_init(self, *args, **kwargs):
    if 'sync_config' in kwargs and isinstance(kwargs['sync_config'], str):
        from ray.train import SyncConfig
        kwargs['sync_config'] = SyncConfig()
    return original_storage_init(self, *args, **kwargs)
train_storage.StorageContext.__init__ = patched_storage_init

# PyTorch weights_only 기본값 변경 우회
original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)
torch.load = patched_torch_load

warnings.filterwarnings('ignore')

print("=" * 80)
print("MultiVAE Multi-layer MLP Architecture Exploration")
print("=" * 80)
print("✅ 라이브러리 로드 완료\n")

# ============================================================
# 1. 디바이스 자동 선택
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

print(f"PyTorch version: {torch.__version__}\n")

# ============================================================
# 2. Ray 초기화
# ============================================================
print("=" * 80)
print("2. Ray 초기화")
print("=" * 80)

import ray

if ray.is_initialized():
    ray.shutdown()
    print("🔄 기존 Ray 인스턴스 종료")

total_cpus = os.cpu_count() or 4

if device == 'cuda':
    num_cpus = total_cpus
    num_gpus = 1
    print(f"🎮 CUDA 모드: CPU {num_cpus}코어, GPU 1개 할당")
elif device == 'mps':
    num_cpus = total_cpus
    num_gpus = 0
    print(f"🍎 MPS 모드: CPU {num_cpus}코어 할당")
else:
    num_cpus = total_cpus
    num_gpus = 0
    print(f"💻 CPU 모드: {num_cpus}코어 할당")

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

print(f"✅ 데이터 로드: {len(df):,}개 상호작용")
print(f"   사용자: {df['user_id'].nunique():,}, 아이템: {df['item_id'].nunique():,}")
print(f"   희소성: {1 - len(df) / (df['user_id'].nunique() * df['item_id'].nunique()):.4%}")

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

print(f"✅ RecBole 데이터셋 생성: {inter_file}\n")

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

# MLP 아키텍처 탐색 공간
# 기존 최적값: latent_dim=128, mlp=[600], dropout=0.549, lr=0.000309
search_space = {
    'latent_dimension': tune.choice([128, 200]),  # 검증된 범위
    'mlp_hidden_size': tune.choice([
        # Single-layer variations
        [400],           # Smaller single layer
        [512],           # Known good
        [600],           # Best from previous
        [800],           # Larger single layer

        # Two-layer decreasing
        [400, 200],      # Small two-layer
        [512, 256],      # Medium two-layer
        [600, 300],      # Large two-layer
        [800, 400],      # Very large two-layer

        # Three-layer
        [600, 400, 200], # Multi-layer hierarchy
    ]),
    'dropout_prob': tune.uniform(0.25, 0.35),  # Near optimal 0.302/0.549
    'anneal_cap': tune.choice([0.2]),          # Lock to optimal
    'learning_rate': tune.loguniform(3e-4, 8e-4),  # Near optimal 0.000309
}

print("✅ 설정 완료")
print(f"   모델: {MODEL_NAME} - MLP Architecture Exploration")
print(f"   타겟: Recall@5")
print(f"\n🔍 탐색 공간:")
print(f"   latent_dimension: [128, 200]")
print(f"   mlp_hidden_size: 9가지 아키텍처")
print(f"     • Single: [400], [512], [600], [800]")
print(f"     • Two-layer: [400,200], [512,256], [600,300], [800,400]")
print(f"     • Three-layer: [600,400,200]")
print(f"   dropout_prob: [0.25, 0.35]")
print(f"   anneal_cap: 0.2 (고정)")
print(f"   learning_rate: [3e-4, 8e-4]\n")

# ============================================================
# 5. Trainable 함수
# ============================================================
print("=" * 80)
print("5. Trainable 함수 정의")
print("=" * 80)

def train_recbole(config_params):
    """Ray Tune trainable - MultiVAE MLP architecture experiment"""
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
print("6. Ray Tune 하이퍼파라미터 최적화")
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
        num_samples=27,  # 9 architectures * 3 trials each
        max_concurrent_trials=max_concurrent_trials,
    ),
    run_config=RunConfig(
        name='recbole_multivae_mlp_arch',
        storage_path=ray_results_path,
    ),
)

print("\n🚀 최적화 시작 (27 trials)...\n")
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

print(f"\n🎯 검증 성능:")
print(f"   Recall@5: {best_metrics['recall@5']:.4f}")
print(f"   NDCG@5: {best_metrics['ndcg@5']:.4f}")

# MLP 아키텍처별 성능 분석
results_df = results.get_dataframe()
print("\n📊 MLP 아키텍처별 평균 성능:")

# mlp_hidden_size별로 그룹화
mlp_performance = {}
for _, row in results_df.iterrows():
    if row['recall@5'] > 0:  # 성공한 trial만
        mlp = str(row['config/mlp_hidden_size'])
        if mlp not in mlp_performance:
            mlp_performance[mlp] = []
        mlp_performance[mlp].append(row['recall@5'])

# 평균 계산 및 정렬
mlp_avg = {k: np.mean(v) for k, v in mlp_performance.items()}
mlp_sorted = sorted(mlp_avg.items(), key=lambda x: x[1], reverse=True)

print("\n아키텍처 | 평균 Recall@5 | 시도 횟수")
print("-" * 50)
for mlp, avg_recall in mlp_sorted:
    count = len(mlp_performance[mlp])
    print(f"{mlp:20s} | {avg_recall:.4f} | {count}회")

# 최적 파라미터 저장
import json
t_params = pd.Timestamp.now()
params_output_dir = f"outputs/{t_params.year}-{t_params.month:02d}-{t_params.day:02d}"
os.makedirs(params_output_dir, exist_ok=True)
params_filename = f"{params_output_dir}/best_hyperparams_multivae_mlp_arch_{t_params.year}{t_params.month:02d}{t_params.day:02d}{t_params.hour:02d}{t_params.minute:02d}{t_params.second:02d}.json"

best_params_to_save = {
    'hyperparameters': {
        'latent_dimension': int(best_config['latent_dimension']),
        'mlp_hidden_size': best_config['mlp_hidden_size'],
        'dropout_prob': float(best_config['dropout_prob']),
        'anneal_cap': float(best_config['anneal_cap']),
        'learning_rate': float(best_config['learning_rate'])
    },
    'validation_metrics': {
        'recall@5': float(best_metrics['recall@5']),
        'ndcg@5': float(best_metrics['ndcg@5']),
        'recall@10': float(best_metrics['recall@10'])
    },
    'mlp_architecture_analysis': {
        arch: {'avg_recall@5': float(avg), 'num_trials': len(mlp_performance[arch])}
        for arch, avg in mlp_sorted
    },
    'timestamp': t_params.strftime('%Y-%m-%d %H:%M:%S'),
    'experiment': 'MLP Architecture Exploration',
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
filename = f"{output_dir}/submit_{MODEL_NAME}_MLP_Arch_{t.year}{t.month:02d}{t.day:02d}{t.hour:02d}{t.minute:02d}{t.second:02d}.csv"

submission.to_csv(filename, index=False)

print(f"\n✅ 제출 파일 생성 완료!")
print(f"   파일: {filename}")
print(f"   검증 Recall@5: {best_valid_score:.4f}")
print(f"   테스트 Recall@5: {test_result['recall@5']:.4f}")

# ============================================================
# 11. 최종 요약
# ============================================================
print("\n" + "=" * 80)
print("MultiVAE MLP Architecture Exploration - 결과 요약")
print("=" * 80)

print(f"\n🤖 실험 정보:")
print(f"   총 trials: {len(results_df)}")
print(f"   성공 trials: {len(results_df[results_df['recall@5'] > 0])}")
print(f"   탐색한 아키텍처: {len(mlp_performance)}가지")

print(f"\n🏆 최고 성능 MLP 아키텍처:")
print(f"   구조: {best_config['mlp_hidden_size']}")
print(f"   Layers: {len(best_config['mlp_hidden_size'])}")
print(f"   검증 Recall@5: {best_valid_result['recall@5']:.4f}")
print(f"   테스트 Recall@5: {test_result['recall@5']:.4f}")

print(f"\n📈 Top 3 MLP 아키텍처:")
for i, (mlp, avg_recall) in enumerate(mlp_sorted[:3], 1):
    print(f"   {i}. {mlp}: {avg_recall:.4f}")

print(f"\n💾 출력 파일:")
print(f"   제출: {filename}")
print(f"   분석: {params_filename}")

print("\n" + "=" * 80)
print("✅ 모든 작업 완료!")
print("=" * 80)

ray.shutdown()
