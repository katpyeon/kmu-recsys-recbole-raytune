#!/usr/bin/env python3
"""
RecBole AutoML - MultiVAE Hyperparameter Optimization with Ray Tune (v5 - Stable Release)

희소성 99.9% 데이터셋에 대한 MultiVAE 하이퍼파라미터 최적화
- 평가 지표: Recall@5
- AutoML: Ray Tune (ASHA Scheduler + Optuna TPE)
- 디바이스: CUDA → MPS → CPU 자동 선택
- 모델: MultiVAE (Variational Autoencoders for Collaborative Filtering)

🔧 v5 개선사항 (성능 하락 문제 해결):
1. 올바른 Core-filtering 적용: Pandas 수동 필터링 대신 RecBole 내장 기능(`user_inter_num_interval`, `item_inter_num_interval`)을 사용하여 데이터 손실 문제를 해결합니다.
2. 안정적인 하이퍼파라미터 공간 복귀: 성능이 가장 좋았던 v1의 탐색 공간으로 되돌려 안정성을 확보합니다.
3. 탐색 횟수 조정: num_samples를 50으로 설정하여 안정적인 탐색을 시도합니다.
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

# Ray Tune imports (최신 API)
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
        return 1  # 문자열이면 기본값 반환
    return original_get_air_verbosity(verbose)
tune_output.get_air_verbosity = patched_get_air_verbosity

# StorageContext __init__ 타입 버그 수정
original_storage_init = train_storage.StorageContext.__init__
def patched_storage_init(self, *args, **kwargs):
    # sync_config가 문자열이면 기본값으로 대체
    if 'sync_config' in kwargs and isinstance(kwargs['sync_config'], str):
        from ray.train import SyncConfig
        kwargs['sync_config'] = SyncConfig()
    return original_storage_init(self, *args, **kwargs)
train_storage.StorageContext.__init__ = patched_storage_init

# PyTorch 2.6+ weights_only 기본값 변경 우회
# RecBole checkpoint 로딩 시 weights_only=False 필요
original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    # weights_only 인자가 명시되지 않은 경우 False로 설정
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)
torch.load = patched_torch_load

warnings.filterwarnings('ignore')

print("=" * 60)
print("RecBole AutoML with Ray Tune - MultiVAE v5")
print("=" * 60)
print("✅ 라이브러리 로드 완료\n")

# ============================================================ 
# 1. 디바이스 자동 선택
# ============================================================ 
print("=" * 60)
print("1. 디바이스 선택")
print("=" * 60)

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
print("=" * 60)
print("2. Ray 초기화")
print("=" * 60)

import ray

if ray.is_initialized():
    ray.shutdown()
    print("🔄 기존 Ray 인스턴스 종료")

# 시스템 CPU 코어 수 자동 감지
total_cpus = os.cpu_count() or 4

# 디바이스별 리소스 할당 전략
if device == 'cuda':
    # CUDA: 전체 CPU 코어 사용 (병렬 처리 최대화)
    num_cpus = total_cpus  # 전체 사용
    num_gpus = 1
    print(f"🎮 CUDA 모드: CPU {num_cpus}/{total_cpus}코어, GPU 1개 할당")
elif device == 'mps':
    # MPS: 통합 메모리로 전체 CPU 사용 가능
    num_cpus = total_cpus
    num_gpus = 0  # MPS는 PyTorch device='mps'로 자동 처리
    print(f"🍎 MPS 모드: CPU {num_cpus}코어 할당 (GPU 자동 사용)")
else:
    # CPU only
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

print("✅ Ray 초기화 완료")
print(f"   사용 가능한 리소스: {ray.available_resources()}\n")

# ============================================================ 
# 3. 데이터 로딩 및 전처리
# ============================================================ 
print("=" * 60)
print("3. 데이터 로딩 및 전처리")
print("=" * 60)

start_time = time.time()

# 데이터 로드
train_file = 'dataset/comp_train.csv'
df = pd.read_csv(train_file)

# BOM 문자 제거
df.columns = [col.replace('\ufeff', '') for col in df.columns]

print(f"✅ 데이터 로드 완료")
print(f"   Total interactions: {len(df):,}")
print(f"   Unique users: {df['user_id'].nunique():,}")
print(f"   Unique items: {df['item_id'].nunique():,}")
print(f"   Sparsity: {1 - len(df) / (df['user_id'].nunique() * df['item_id'].nunique()):.4%}")


# RecBole 형식으로 변환 (Pandas 필터링 제거)
df_recbole = pd.DataFrame({
    'user_id:token': df['user_id'],
    'item_id:token': df['item_id'],
    'rating:float': 1.0
})

# RecBole 데이터셋 디렉토리 생성
dataset_dir = 'dataset/kaggle_recsys'
os.makedirs(dataset_dir, exist_ok=True)

# .inter 파일로 저장
inter_file = os.path.join(dataset_dir, 'kaggle_recsys.inter')
df_recbole.to_csv(inter_file, sep='\t', index=False)

print(f"\n✅ RecBole 데이터셋 생성 완료")
print(f"   파일: {inter_file}")
print(f"   형식: Tab-separated (.inter)")
print(f"   소요 시간: {time.time() - start_time:.2f}초\n")

# ============================================================ 
# 4. Ray Tune 설정
# ============================================================ 
print("=" * 60)
print("4. Ray Tune 설정")
print("=" * 60)

MODEL_NAME = 'MultiVAE'
MIN_INTERACTIONS = 5

# 절대 경로로 데이터셋 경로 설정 (Ray Tune 병렬 실행 시 필요)
DATASET_PATH = str(Path(__file__).parent / 'dataset')

# 디바이스별 배치 크기 설정
if device == 'cuda':
    # CUDA: GPU 메모리 활용 최대화 (최적화)
    train_batch_size = 4096
    eval_batch_size = 102400  # 평가는 결과에 영향 없으므로 최대로 설정
    print(f"🎮 CUDA 배치 크기: train={train_batch_size}, eval={eval_batch_size}")
elif device == 'mps':
    # MPS: 통합 메모리로 큰 배치 크기 사용 가능
    train_batch_size = 2048
    eval_batch_size = 4096
    print(f"🍎 MPS 배치 크기: train={train_batch_size}, eval={eval_batch_size}")
else:
    # CPU: 메모리 여유 있음
    train_batch_size = 2048
    eval_batch_size = 4096
    print(f"💻 CPU 배치 크기: train={train_batch_size}, eval={eval_batch_size}")

base_config = {
    'data_path': DATASET_PATH,
    'dataset': 'kaggle_recsys',
    'USER_ID_FIELD': 'user_id',
    'ITEM_ID_FIELD': 'item_id',
    'RATING_FIELD': 'rating',
    'load_col': {'inter': ['user_id', 'item_id', 'rating']},
    
    # --- RecBole 내장 Core-filtering 적용 (v5 개선) ---
    'user_inter_num_interval': f'[{MIN_INTERACTIONS}, inf)',
    'item_inter_num_interval': f'[{MIN_INTERACTIONS}, inf)',

    'train_neg_sample_args': None,  # 필수! MultiVAE는 non-sampling 모델
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
    'reproducibility': False,  # cuDNN benchmark 활성화 (최적화)
    'show_progress': False,
    # ===== 성능 최적화 설정 =====
    'worker': 4,           # DataLoader 병렬 처리 (CPU 데이터 로딩 가속)
}

# 하이퍼파라미터 탐색 공간 (v1, 안정적인 버전으로 복귀)
search_space = {
    'latent_dimension': tune.choice([128, 200, 256]),
    'mlp_hidden_size': tune.choice([[600], [512]]),
    'dropout_prob': tune.uniform(0.3, 0.7),
    'anneal_cap': tune.choice([0.1, 0.2, 0.3]),
    'learning_rate': tune.loguniform(1e-4, 1e-2),
}

print(f"✅ 기본 설정 완료")
print(f"   모델: {MODEL_NAME} v5 (Stable Release)")
print(f"   타겟 메트릭: Recall@5")
print(f"   디바이스: {device}")
print(f"\n✨ v5 개선사항:")
print(f"   • RecBole Core-filtering: min_interactions={MIN_INTERACTIONS}")
print(f"   • 안정적인 탐색 공간으로 복귀 (v1 기준)")
print(f"\n⚡ 성능 최적화 적용:")
print(f"   • 배치 크기: train={train_batch_size}, eval={eval_batch_size}")
print(f"   • DataLoader workers: 4 (병렬 데이터 로딩)")
print(f"   • cuDNN benchmark: 활성화 (1.3~1.7배 가속)")
print(f"\n🔍 하이퍼파라미터 탐색 공간 (v1 복귀):")
print(f"   latent_dimension: [128, 200, 256]")
print(f"   mlp_hidden_size: [[600], [512]]")
print(f"   dropout_prob: [0.3, 0.7]")
print(f"   anneal_cap: [0.1, 0.2, 0.3]")
print(f"   learning_rate: [1e-4, 1e-2]\n")


# ============================================================ 
# 5. Trainable 함수 정의
# ============================================================ 
print("=" * 60)
print("5. Trainable 함수 정의")
print("=" * 60)

def train_recbole(config_params):
    """Ray Tune trainable 함수 - RecBole MultiVAE 학습"""
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

        recall_5 = best_valid_result['recall@5']
        ndcg_5 = best_valid_result['ndcg@5']
        recall_10 = best_valid_result['recall@10']

        train.report({
            'recall@5': recall_5,
            'ndcg@5': ndcg_5,
            'recall@10': recall_10,
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
# 6. Ray Tune AutoML 실행
# ============================================================ 
print("=" * 60)
print("6. Ray Tune 하이퍼파라미터 최적화 시작")
print("=" * 60)

start_time = time.time()

# ASHA Scheduler 설정
scheduler = ASHAScheduler(
    metric='recall@5',
    mode='max',
    max_t=100,
    grace_period=10,
    reduction_factor=2,
)

# Optuna Search
search_alg = OptunaSearch(
    metric='recall@5',
    mode='max',
)

# Ray Tune 저장 경로
ray_results_path = str(Path('./ray_results').resolve())
print(f"📁 Ray Tune 결과 저장 경로: {ray_results_path}")

# 디바이스별 Trial 리소스 할당
if device == 'cuda':
    # CUDA: GPU 메모리 활용 최적화 (6개 동시 실행으로 전체 시간 절반 단축)
    resources_per_trial = {"cpu": 1, "gpu": 0.16}
    max_concurrent_trials = 6
    print(f"\n🎮 CUDA Trial 설정 (최적화):")
    print(f"   Trial당 리소스: CPU 1코어, GPU 0.16개")
    print(f"   최대 동시 실행: {max_concurrent_trials}개")
    print(f"   → 병렬 처리 최대화, 전체 AutoML 시간 2배 단축")
elif device == 'mps':
    # MPS: 통합 메모리로 제한 불필요
    resources_per_trial = {"cpu": 2}
    max_concurrent_trials = None  # Ray가 자동 결정
    print(f"\n🍎 MPS Trial 설정:")
    print(f"   Trial당 리소스: CPU 2코어")
    print(f"   최대 동시 실행: 제한 없음 (자동)")
    print(f"   → 통합 메모리로 병렬 실행 최적화")
else:
    # CPU only
    resources_per_trial = {"cpu": 2}
    max_concurrent_trials = None
    print(f"\n💻 CPU Trial 설정:")
    print(f"   Trial당 리소스: CPU 2코어")
    print(f"   최대 동시 실행: 제한 없음 (자동)")

# Ray Tune 실행
tuner = tune.Tuner(
    tune.with_resources(train_recbole, resources=resources_per_trial),
    param_space=search_space,
    tune_config=tune.TuneConfig(
        scheduler=scheduler,
        search_alg=search_alg,
        num_samples=50,  # ✨ v5: 탐색 횟수 조정
        max_concurrent_trials=max_concurrent_trials,
    ),
    run_config=RunConfig(
        name='recbole_multivae_automl_v5', # v5 버전으로 이름 변경
        storage_path=ray_results_path,
    ),
)

print("\n🚀 최적화 시작...\n")
results = tuner.fit()

print("\n" + "=" * 60)
print("✅ Ray Tune 최적화 완료")
print(f"   소요 시간: {time.time() - start_time:.2f}초")
print("=" * 60 + "\n")

# ============================================================ 
# 7. 최적 하이퍼파라미터 추출
# ============================================================ 
print("=" * 60)
print("7. 최적 하이퍼파라미터 추출")
print("=" * 60)

best_result = results.get_best_result(metric='recall@5', mode='max')
best_config = best_result.config
best_metrics = best_result.metrics

print("\n🏆 최적 하이퍼파라미터:")
print(f"   latent_dimension: {int(best_config['latent_dimension'])}")
print(f"   mlp_hidden_size: {best_config['mlp_hidden_size']}")
print(f"   dropout_prob: {best_config['dropout_prob']:.4f}")
print(f"   anneal_cap: {best_config['anneal_cap']:.4f}")
print(f"   learning_rate: {best_config['learning_rate']:.6f}")

print(f"\n🎯 최고 검증 성능:")
print(f"   Recall@5: {best_metrics['recall@5']:.4f}")
print(f"   NDCG@5: {best_metrics['ndcg@5']:.4f}")
print(f"   Recall@10: {best_metrics['recall@10']:.4f}")

# 최적 하이퍼파라미터 JSON 파일로 저장 (재현성 확보)
import json
t_params = pd.Timestamp.now()
params_output_dir = f"outputs/{t_params.year}-{t_params.month:02d}-{t_params.day:02d}"
os.makedirs(params_output_dir, exist_ok=True)
params_filename = f"{params_output_dir}/best_hyperparams_multivae_v5_{t_params.year}{t_params.month:02d}{t_params.day:02d}{t_params.hour:02d}{t_params.minute:02d}{t_params.second:02d}.json"

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
    'timestamp': t_params.strftime('%Y-%m-%d %H:%M:%S'),
    'model': MODEL_NAME + "_v5", # v5 버전으로 모델 이름 변경
    'device': device,
    'num_trials': len(results.get_dataframe())
}

with open(params_filename, 'w') as f:
    json.dump(best_params_to_save, f, indent=2)

print(f"\n💾 최적 파라미터 저장 완료: {params_filename}\n")

# ============================================================ 
# 8. 최적 모델로 최종 학습
# ============================================================ 
print("=" * 60)
print("8. 최적 모델로 최종 학습")
print("=" * 60)

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

print("\n✅ 최종 모델 학습 완료")
print(f"   소요 시간: {time.time() - start_time:.2f}초")
print(f"\n📊 검증 성능:")
print(f"   Recall@5:  {best_valid_result['recall@5']:.4f}")
print(f"   NDCG@5:    {best_valid_result['ndcg@5']:.4f}")
print(f"   Recall@10: {best_valid_result['recall@10']:.4f}")
print(f"\n📊 테스트 성능:")
print(f"   Recall@5:  {test_result['recall@5']:.4f}")
print(f"   NDCG@5:    {test_result['ndcg@5']:.4f}")
print(f"   Recall@10: {test_result['recall@10']:.4f}\n")

# ============================================================ 
# 9. 전체 사용자 추천 생성
# ============================================================ 
print("=" * 60)
print("9. 전체 사용자 추천 생성")
print("=" * 60)

start_time = time.time()

all_users = dataset.inter_feat['user_id'].unique()
all_recommendations = {}

topk = 10

model.eval()
with torch.no_grad():
    for i, user_id in enumerate(all_users):
        user_external = dataset.id2token('user_id', user_id.item())

        # full_sort_topk()를 사용하여 추천 생성 (내부 ID 전달)
        topk_scores, topk_indices = full_sort_topk(
            [user_id.item()],  # 내부 ID (정수) 전달
            model,
            test_data,
            k=topk,
            device=config['device']
        )

        # 추천 아이템 외부 ID로 변환
        topk_items_internal = topk_indices[0].cpu().tolist()
        items_external = [dataset.id2token('item_id', int(item)) for item in topk_items_internal]
        all_recommendations[user_external] = items_external

        if (i + 1) % 1000 == 0:
            print(f"   진행: {i + 1}/{len(all_users)} 사용자 처리 완료")

print(f"\n✅ 추천 생성 완료")
print(f"   총 사용자 수: {len(all_recommendations):,}")
print(f"   사용자당 추천 수: {topk}")
print(f"   소요 시간: {time.time() - start_time:.2f}초\n")

# ============================================================ 
# 10. 제출 파일 생성
# ============================================================ 
print("=" * 60)
print("10. 제출 파일 생성")
print("=" * 60)

start_time = time.time()

submission = pd.DataFrame([
    {'user_id': user_id, 'item_ids': ' '.join(recs)}
    for user_id, recs in all_recommendations.items()
])

print(f"✅ 제출 데이터 변환 완료")
print(f"   총 행 수 (사용자 수): {len(submission):,}")
print(f"   사용자당 추천 수: {topk}")

t = pd.Timestamp.now()
output_dir = f"outputs/{t.year}-{t.month:02d}-{t.day:02d}"
os.makedirs(output_dir, exist_ok=True)
filename = f"{output_dir}/submit_{MODEL_NAME}_v5_RayTune_{t.year}{t.month:02d}{t.day:02d}{t.hour:02d}{t.minute:02d}{t.second:02d}.csv" # v5 버전으로 파일명 변경

submission.to_csv(filename, index=False)

print(f"\n" + "=" * 60)
print(f"✅ 제출 파일 생성 완료!")
print("=" * 60)
print(f"파일명: {filename}")
print(f"검증 Recall@5: {best_valid_score:.4f}")
print(f"테스트 Recall@5: {test_result['recall@5']:.4f}")
print(f"소요 시간: {time.time() - start_time:.2f}초")
print("=" * 60)

# ============================================================ 
# 11. 결과 요약
# ============================================================ 
print("\n" + "=" * 60)
print("Ray Tune AutoML 최적화 결과 요약 - MultiVAE v5")
print("=" * 60)

results_df = results.get_dataframe()

print(f"\n📊 데이터셋 정보 (필터링 적용됨):")
print(f"   RecBole이 자동으로 필터링한 사용자/아이템/상호작용 수를 사용합니다.")


print(f"\n🤖 AutoML 정보:")
print(f"   모델: {MODEL_NAME}_v5") # v5 버전으로 모델 이름 변경
print(f"   디바이스: {device}")
print(f"   AutoML 방식: Ray Tune (ASHA + Optuna TPE)")
print(f"   총 시도 횟수: {len(results_df)}")
print(f"   완료된 trial: {len(results_df[results_df['recall@5'] > 0])}")

print(f"\n🏆 최적 하이퍼파라미터:")
print(f"   latent_dimension: {int(best_config['latent_dimension'])}")
print(f"   mlp_hidden_size: {best_config['mlp_hidden_size']}")
print(f"   dropout_prob: {best_config['dropout_prob']:.4f}")
print(f"   anneal_cap: {best_config['anneal_cap']:.4f}")
print(f"   learning_rate: {best_config['learning_rate']:.6f}")

print(f"\n📈 최종 성능:")
print(f"   검증 Recall@5: {best_valid_result['recall@5']:.4f}")
print(f"   테스트 Recall@5: {test_result['recall@5']:.4f}")
print(f"   검증 NDCG@5: {best_valid_result['ndcg@5']:.4f}")
print(f"   테스트 NDCG@5: {test_result['ndcg@5']:.4f}")

print(f"\n💾 출력 파일:")
print(f"   제출 파일: {filename}")
print(f"   최적 파라미터: {params_filename}")
print(f"   Ray Tune 결과: {ray_results_path}/recbole_multivae_automl_v5/") # v5 버전으로 경로 변경

print("\n📊 상위 5개 Trial 결과:")
top5 = results_df.nlargest(5, 'recall@5')[['config/latent_dimension', 'config/mlp_hidden_size',
                                              'config/dropout_prob', 'config/anneal_cap',
                                              'config/learning_rate', 'recall@5', 'ndcg@5']]
print(top5.to_string(index=False))

print("\n" + "=" * 60)
print("✅ 모든 작업 완료!")
print("=" * 60)

# Ray 종료
ray.shutdown()
print("\n✅ Ray 종료 완료")
