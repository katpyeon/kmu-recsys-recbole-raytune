#!/usr/bin/env python3
"""
RecBole AutoML - EASE Hyperparameter Optimization with Ray Tune

희소성 99.9% 데이터셋에 대한 EASE 하이퍼파라미터 최적화
- 평가 지표: Recall@5
- AutoML: Ray Tune (ASHA Scheduler + Optuna TPE)
- 디바이스: CUDA → MPS → CPU 자동 선택
- 모델: EASE (Embarrassingly Shallow Autoencoders)
- 특징: Closed-form solution, 단일 하이퍼파라미터(reg_weight)
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
print("RecBole AutoML with Ray Tune - EASE")
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

MODEL_NAME = 'EASE'

# 절대 경로로 데이터셋 경로 설정 (Ray Tune 병렬 실행 시 필요)
DATASET_PATH = str(Path(__file__).parent / 'dataset')

# EASE는 GPU 사용 안함 (행렬 계산만 수행)
# 배치 크기 설정도 불필요 (반복 학습 없음)

base_config = {
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
    'metrics': ['Recall', 'NDCG', 'MRR'],
    'topk': [5, 10, 20],
    'valid_metric': 'Recall@5',
    'device': 'cpu',  # EASE는 CPU만 사용
    'seed': 2024,
    'reproducibility': True,
    'show_progress': False,
}

# 하이퍼파라미터 탐색 공간 (RecBole 문서 + EASE 논문 기반)
# 출처: Steck, "Embarrassingly Shallow Autoencoders for Sparse Data", WWW 2019
# EASE는 단 하나의 하이퍼파라미터만 튜닝 (reg_weight)
search_space = {
    'reg_weight': tune.choice([10.0, 100.0, 250.0, 500.0, 1000.0]),  # 기본: 250
}

print(f"✅ 기본 설정 완료")
print(f"   모델: {MODEL_NAME}")
print(f"   타겟 메트릭: Recall@5")
print(f"   디바이스: CPU (EASE는 GPU 미사용)")
print(f"\n🔍 하이퍼파라미터 탐색 공간 (RecBole 문서 + 논문 기반):")
print(f"   reg_weight: [10, 100, 250, 500, 1000] (기본: 250)")
print(f"   ⚡ EASE는 단일 하이퍼파라미터 - 매우 빠른 최적화!")
print(f"   ⚡ Closed-form solution - 반복 학습 불필요\n")

# ============================================================
# 5. Trainable 함수 정의
# ============================================================
print("=" * 60)
print("5. Trainable 함수 정의")
print("=" * 60)

def train_recbole(config_params):
    """Ray Tune trainable 함수 - RecBole EASE 학습"""
    from ray import train

    config_dict = base_config.copy()
    config_dict.update({
        'model': MODEL_NAME,
        'reg_weight': config_params['reg_weight'],
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

# ASHA Scheduler 설정 (EASE는 단일 epoch이므로 효과 제한적)
scheduler = ASHAScheduler(
    metric='recall@5',
    mode='max',
    max_t=1,  # EASE는 epoch 개념 없음
    grace_period=1,
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

# EASE는 CPU만 사용, 빠른 실행
resources_per_trial = {"cpu": 2}
max_concurrent_trials = None  # 제한 없음

print(f"\n💻 EASE Trial 설정:")
print(f"   Trial당 리소스: CPU 2코어")
print(f"   최대 동시 실행: 제한 없음 (자동)")
print(f"   → Closed-form solution으로 매우 빠름")

# Ray Tune 실행
tuner = tune.Tuner(
    tune.with_resources(train_recbole, resources=resources_per_trial),
    param_space=search_space,
    tune_config=tune.TuneConfig(
        scheduler=scheduler,
        search_alg=search_alg,
        num_samples=10,  # EASE는 파라미터 1개이므로 10번이면 충분
        max_concurrent_trials=max_concurrent_trials,
    ),
    run_config=RunConfig(
        name='recbole_ease_automl',
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
print(f"   reg_weight: {best_config['reg_weight']:.1f}")

print(f"\n🎯 최고 검증 성능:")
print(f"   Recall@5: {best_metrics['recall@5']:.4f}")
print(f"   NDCG@5: {best_metrics['ndcg@5']:.4f}")
print(f"   Recall@10: {best_metrics['recall@10']:.4f}")

# 최적 하이퍼파라미터 JSON 파일로 저장 (재현성 확보)
import json
t_params = pd.Timestamp.now()
params_output_dir = f"outputs/{t_params.year}-{t_params.month:02d}-{t_params.day:02d}"
os.makedirs(params_output_dir, exist_ok=True)
params_filename = f"{params_output_dir}/best_hyperparams_ease_{t_params.year}{t_params.month:02d}{t_params.day:02d}{t_params.hour:02d}{t_params.minute:02d}{t_params.second:02d}.json"

best_params_to_save = {
    'hyperparameters': {
        'reg_weight': float(best_config['reg_weight'])
    },
    'validation_metrics': {
        'recall@5': float(best_metrics['recall@5']),
        'ndcg@5': float(best_metrics['ndcg@5']),
        'recall@10': float(best_metrics['recall@10'])
    },
    'timestamp': t_params.strftime('%Y-%m-%d %H:%M:%S'),
    'model': MODEL_NAME,
    'device': 'cpu',
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
    'reg_weight': best_config['reg_weight'],
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
print("Ray Tune AutoML 최적화 결과 요약 - EASE")
print("=" * 60)

results_df = results.get_dataframe()

print(f"\n📊 데이터셋 정보:")
print(f"   사용자 수: {df['resume_seq'].nunique():,}")
print(f"   아이템 수: {df['recruitment_seq'].nunique():,}")
print(f"   상호작용 수: {len(df):,}")
print(f"   희소성: {1 - len(df) / (df['resume_seq'].nunique() * df['recruitment_seq'].nunique()):.4%}")

print(f"\n🤖 AutoML 정보:")
print(f"   모델: {MODEL_NAME}")
print(f"   디바이스: CPU (EASE는 GPU 미사용)")
print(f"   AutoML 방식: Ray Tune (ASHA + Optuna TPE)")
print(f"   총 시도 횟수: {len(results_df)}")
print(f"   완료된 trial: {len(results_df[results_df['recall@5'] > 0])}")

print(f"\n🏆 최적 하이퍼파라미터:")
print(f"   reg_weight: {best_config['reg_weight']:.1f}")

print(f"\n📈 최종 성능:")
print(f"   검증 Recall@5: {best_valid_result['recall@5']:.4f}")
print(f"   테스트 Recall@5: {test_result['recall@5']:.4f}")
print(f"   검증 NDCG@5: {best_valid_result['ndcg@5']:.4f}")
print(f"   테스트 NDCG@5: {test_result['ndcg@5']:.4f}")

print(f"\n💾 출력 파일:")
print(f"   제출 파일: {filename}")
print(f"   최적 파라미터: {params_filename}")
print(f"   Ray Tune 결과: {ray_results_path}/recbole_ease_automl/")

print("\n📊 모든 Trial 결과:")
trial_results = results_df[['config/reg_weight', 'recall@5', 'ndcg@5']].sort_values('recall@5', ascending=False)
print(trial_results.to_string(index=False))

print("\n" + "=" * 60)
print("✅ 모든 작업 완료!")
print("=" * 60)

# Ray 종료
ray.shutdown()
print("\n✅ Ray 종료 완료")
