#!/usr/bin/env python3
"""
RecBole AutoML - RecVAE Hyperparameter Optimization with Ray Tune

희소성 99.9% 데이터셋에 대한 RecVAE 하이퍼파라미터 최적화
- 평가 지표: Recall@5
- AutoML: Ray Tune (ASHA Scheduler + Optuna TPE)
- 디바이스: CUDA → MPS → CPU 자동 선택
- 모델: RecVAE (Collaborative Filtering with Recurrent Variational Autoencoders)
- 특징: Composite Prior, 적응형 β, 교대 학습
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

warnings.filterwarnings('ignore')

print("=" * 60)
print("RecBole AutoML with Ray Tune - RecVAE")
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
    # CUDA: CPU/GPU 경합 방지 위해 코어 수 제한
    num_cpus = total_cpus // 2  # 절반만 사용 (열 관리)
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
train_file = 'dataset/apply_train.csv'
df = pd.read_csv(train_file)

# BOM 문자 제거
df.columns = [col.replace('\ufeff', '') for col in df.columns]

print(f"✅ 데이터 로드 완료")
print(f"   Total interactions: {len(df):,}")
print(f"   Unique users: {df['resume_seq'].nunique():,}")
print(f"   Unique items: {df['recruitment_seq'].nunique():,}")
print(f"   Sparsity: {1 - len(df) / (df['resume_seq'].nunique() * df['recruitment_seq'].nunique()):.4%}")

# RecBole 형식으로 변환
df_recbole = pd.DataFrame({
    'user_id:token': df['resume_seq'],
    'item_id:token': df['recruitment_seq'],
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

MODEL_NAME = 'RecVAE'

# 절대 경로로 데이터셋 경로 설정 (Ray Tune 병렬 실행 시 필요)
DATASET_PATH = str(Path(__file__).parent / 'dataset')

# 디바이스별 배치 크기 설정
if device == 'cuda':
    # CUDA: GPU 메모리 경합 방지 위해 배치 크기 축소
    train_batch_size = 1024
    eval_batch_size = 2048
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
    'train_neg_sample_args': None,  # 필수! RecVAE는 non-sampling 모델
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
    'reproducibility': True,
    'show_progress': False,
}

# 하이퍼파라미터 탐색 공간 (RecBole 문서 + RecVAE 논문 기반)
# 출처: Shenbin et al., "RecVAE: A New Variational Autoencoder for Top-N Recommendations", WSDM 2020
search_space = {
    'hidden_dimension': tune.choice([512, 600]),           # 오토인코더 은닉층, 기본: 600
    'latent_dimension': tune.choice([128, 200, 256]),      # 잠재 공간 차원, 기본: 200
    'dropout_prob': tune.uniform(0.3, 0.7),                # 드롭아웃 확률, 기본: 0.5
    'gamma': tune.loguniform(0.001, 0.01),                 # 적응형 β 계산용, 기본: 0.005
    'n_enc_epochs': tune.choice([1, 3]),                   # 인코더 학습 반복, 기본: 3
    'learning_rate': tune.loguniform(1e-4, 1e-2),
}

print(f"✅ 기본 설정 완료")
print(f"   모델: {MODEL_NAME}")
print(f"   타겟 메트릭: Recall@5")
print(f"   디바이스: {device}")
print(f"\n🔍 하이퍼파라미터 탐색 공간 (RecBole 문서 + 논문 기반):")
print(f"   hidden_dimension: [512, 600] (기본: 600)")
print(f"   latent_dimension: [128, 200, 256] (기본: 200)")
print(f"   dropout_prob: [0.3, 0.7] (기본: 0.5)")
print(f"   gamma: [0.001, 0.01] (적응형 β, 기본: 0.005)")
print(f"   n_enc_epochs: [1, 3] (인코더 학습 반복, 기본: 3)")
print(f"   learning_rate: [1e-4, 1e-2] (log-uniform)")
print(f"\n💡 RecVAE 특징: Composite Prior, 적응형 β, 교대 학습\n")

# ============================================================
# 5. Trainable 함수 정의
# ============================================================
print("=" * 60)
print("5. Trainable 함수 정의")
print("=" * 60)

def train_recbole(config_params):
    """Ray Tune trainable 함수 - RecBole RecVAE 학습"""
    from ray import train

    config_dict = base_config.copy()
    config_dict.update({
        'model': MODEL_NAME,
        'hidden_dimension': int(config_params['hidden_dimension']),
        'latent_dimension': int(config_params['latent_dimension']),
        'dropout_prob': config_params['dropout_prob'],
        'gamma': config_params['gamma'],
        'n_enc_epochs': int(config_params['n_enc_epochs']),
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
    # CUDA: GPU 메모리 경합 방지 위해 동시 실행 제한
    resources_per_trial = {"cpu": 4, "gpu": 0.5}
    max_concurrent_trials = 2
    print(f"\n🎮 CUDA Trial 설정:")
    print(f"   Trial당 리소스: CPU 4코어, GPU 0.5개")
    print(f"   최대 동시 실행: {max_concurrent_trials}개")
    print(f"   → GPU 메모리 경합 방지, CPU 열 관리")
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
        num_samples=30,  # RecVAE 탐색 공간에 맞게 설정
        max_concurrent_trials=max_concurrent_trials,
    ),
    run_config=RunConfig(
        name='recbole_recvae_automl',
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
print(f"   hidden_dimension: {int(best_config['hidden_dimension'])}")
print(f"   latent_dimension: {int(best_config['latent_dimension'])}")
print(f"   dropout_prob: {best_config['dropout_prob']:.4f}")
print(f"   gamma: {best_config['gamma']:.6f}")
print(f"   n_enc_epochs: {int(best_config['n_enc_epochs'])}")
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
params_filename = f"{params_output_dir}/best_hyperparams_recvae_{t_params.year}{t_params.month:02d}{t_params.day:02d}{t_params.hour:02d}{t_params.minute:02d}{t_params.second:02d}.json"

best_params_to_save = {
    'hyperparameters': {
        'hidden_dimension': int(best_config['hidden_dimension']),
        'latent_dimension': int(best_config['latent_dimension']),
        'dropout_prob': float(best_config['dropout_prob']),
        'gamma': float(best_config['gamma']),
        'n_enc_epochs': int(best_config['n_enc_epochs']),
        'learning_rate': float(best_config['learning_rate'])
    },
    'validation_metrics': {
        'recall@5': float(best_metrics['recall@5']),
        'ndcg@5': float(best_metrics['ndcg@5']),
        'recall@10': float(best_metrics['recall@10'])
    },
    'timestamp': t_params.strftime('%Y-%m-%d %H:%M:%S'),
    'model': MODEL_NAME,
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
    'hidden_dimension': int(best_config['hidden_dimension']),
    'latent_dimension': int(best_config['latent_dimension']),
    'dropout_prob': best_config['dropout_prob'],
    'gamma': best_config['gamma'],
    'n_enc_epochs': int(best_config['n_enc_epochs']),
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

topk = 5

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

result = []
for user_id, recs in all_recommendations.items():
    for item_id in recs:
        result.append((user_id, item_id))

submission = pd.DataFrame(result, columns=['resume_seq', 'recruitment_seq'])

print(f"✅ 제출 데이터 변환 완료")
print(f"   총 행 수: {len(submission):,}")
print(f"   예상 추천 수/사용자: {len(submission) / len(all_recommendations):.2f}")

t = pd.Timestamp.now()
output_dir = f"outputs/{t.year}-{t.month:02d}-{t.day:02d}"
os.makedirs(output_dir, exist_ok=True)
filename = f"{output_dir}/submit_{MODEL_NAME}_RayTune_{t.year}{t.month:02d}{t.day:02d}{t.hour:02d}{t.minute:02d}{t.second:02d}.csv"

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
print("Ray Tune AutoML 최적화 결과 요약 - RecVAE")
print("=" * 60)

results_df = results.get_dataframe()

print(f"\n📊 데이터셋 정보:")
print(f"   사용자 수: {df['resume_seq'].nunique():,}")
print(f"   아이템 수: {df['recruitment_seq'].nunique():,}")
print(f"   상호작용 수: {len(df):,}")
print(f"   희소성: {1 - len(df) / (df['resume_seq'].nunique() * df['recruitment_seq'].nunique()):.4%}")

print(f"\n🤖 AutoML 정보:")
print(f"   모델: {MODEL_NAME}")
print(f"   디바이스: {device}")
print(f"   AutoML 방식: Ray Tune (ASHA + Optuna TPE)")
print(f"   총 시도 횟수: {len(results_df)}")
print(f"   완료된 trial: {len(results_df[results_df['recall@5'] > 0])}")

print(f"\n🏆 최적 하이퍼파라미터:")
print(f"   hidden_dimension: {int(best_config['hidden_dimension'])}")
print(f"   latent_dimension: {int(best_config['latent_dimension'])}")
print(f"   dropout_prob: {best_config['dropout_prob']:.4f}")
print(f"   gamma: {best_config['gamma']:.6f}")
print(f"   n_enc_epochs: {int(best_config['n_enc_epochs'])}")
print(f"   learning_rate: {best_config['learning_rate']:.6f}")

print(f"\n📈 최종 성능:")
print(f"   검증 Recall@5: {best_valid_result['recall@5']:.4f}")
print(f"   테스트 Recall@5: {test_result['recall@5']:.4f}")
print(f"   검증 NDCG@5: {best_valid_result['ndcg@5']:.4f}")
print(f"   테스트 NDCG@5: {test_result['ndcg@5']:.4f}")

print(f"\n💾 출력 파일:")
print(f"   제출 파일: {filename}")
print(f"   최적 파라미터: {params_filename}")
print(f"   Ray Tune 결과: {ray_results_path}/recbole_recvae_automl/")

print("\n📊 상위 5개 Trial 결과:")
top5 = results_df.nlargest(5, 'recall@5')[['config/hidden_dimension', 'config/latent_dimension',
                                              'config/dropout_prob', 'config/gamma',
                                              'config/n_enc_epochs', 'config/learning_rate',
                                              'recall@5', 'ndcg@5']]
print(top5.to_string(index=False))

print("\n" + "=" * 60)
print("✅ 모든 작업 완료!")
print("=" * 60)

# Ray 종료
ray.shutdown()
print("\n✅ Ray 종료 완료")
