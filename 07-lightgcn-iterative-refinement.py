#!/usr/bin/env python3
"""
LightGCN 실험 1: 반복적 탐색 범위 좁히기 (Iterative Refinement)

Baseline: Recall@5 = 0.0882
최적 파라미터: embedding_size=256, n_layers=1, reg_weight=4e-5, lr=0.004

전략:
- Iteration 1: 최적값 주변 정밀 탐색
- Iteration 2: 더 좁은 범위로 집중 탐색
- 자동 중단: 연속 개선 없으면 중지
"""

import os, sys, warnings, pandas as pd, numpy as np, time
from datetime import datetime
from pathlib import Path
import torch
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_seed, get_model, get_trainer
from recbole.utils.case_study import full_sort_topk
from ray import tune
from ray.train import RunConfig
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch

# Ray Tune patches
import ray.tune.experimental.output as tune_output
import ray.train._internal.storage as train_storage

original_get_air_verbosity = tune_output.get_air_verbosity
def patched_get_air_verbosity(verbose):
    return 1 if isinstance(verbose, str) else original_get_air_verbosity(verbose)
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
print("LightGCN 반복적 정밀 탐색 (Iterative Refinement)")
print("=" * 80)

# Device
if torch.cuda.is_available():
    device = 'cuda'
    print(f"🚀 CUDA ({torch.cuda.get_device_name(0)})")
else:
    device = 'cpu'
    print(f"💻 CPU")

# Ray init
import ray
if ray.is_initialized():
    ray.shutdown()

num_cpus = os.cpu_count() or 4
num_gpus = 1 if device == 'cuda' else 0
ray.init(ignore_reinit_error=True, include_dashboard=False, num_cpus=num_cpus, num_gpus=num_gpus, _temp_dir=None, _metrics_export_port=None, configure_logging=False)
print("✅ Ray 초기화\n")

# Data
train_file = 'dataset/comp_train.csv'
df = pd.read_csv(train_file)
df.columns = [col.replace('\ufeff', '') for col in df.columns]
print(f"✅ 데이터: {len(df):,}개, 사용자: {df['user_id'].nunique():,}, 아이템: {df['item_id'].nunique():,}\n")

df_recbole = pd.DataFrame({'user_id:token': df['user_id'], 'item_id:token': df['item_id'], 'rating:float': 1.0})
dataset_dir = 'dataset/kaggle_recsys'
os.makedirs(dataset_dir, exist_ok=True)
inter_file = os.path.join(dataset_dir, 'kaggle_recsys.inter')
df_recbole.to_csv(inter_file, sep='\t', index=False)

MODEL_NAME = 'LightGCN'
DATASET_PATH = str(Path(__file__).parent / 'dataset')
BASELINE_RECALL = 0.0882

base_config = {
    'data_path': DATASET_PATH,
    'dataset': 'kaggle_recsys',
    'USER_ID_FIELD': 'user_id',
    'ITEM_ID_FIELD': 'item_id',
    'RATING_FIELD': 'rating',
    'load_col': {'inter': ['user_id', 'item_id', 'rating']},
    'eval_args': {'split': {'RS': [0.8, 0.1, 0.1]}, 'order': 'RO', 'mode': 'full', 'group_by': 'user'},
    'metrics': ['Recall', 'NDCG', 'MRR'],
    'topk': [5, 10, 20],
    'valid_metric': 'Recall@5',
    'device': device,
    'epochs': 500,
    'stopping_step': 10,
    'train_batch_size': 4096 if device == 'cuda' else 2048,
    'eval_batch_size': 102400 if device == 'cuda' else 4096,
    'seed': 2024,
    'reproducibility': False,
    'show_progress': False,
    'worker': 4,
}

def get_search_space(iteration, iterations):
    """반복별 탐색 공간"""
    if iteration == 1:
        # Iteration 1: 최적값 주변 정밀 탐색
        return {
            'embedding_size': tune.choice([200, 256, 320]),  # 256 중심
            'n_layers': tune.choice([1, 2]),  # 1이 최적이었지만 2도 재확인
            'reg_weight': tune.loguniform(2e-5, 8e-5),  # 4e-5 중심
            'learning_rate': tune.loguniform(0.003, 0.005),  # 0.004 중심
        }
    else:
        # Iteration 2+: 이전 최고 결과 기반 더 좁은 범위
        prev_best = max(iterations, key=lambda x: x['recall@5'])
        best_emb = prev_best['config']['embedding_size']
        best_lr = prev_best['config']['learning_rate']
        best_reg = prev_best['config']['reg_weight']

        return {
            'embedding_size': tune.choice([best_emb]),  # 고정
            'n_layers': tune.choice([prev_best['config']['n_layers']]),  # 고정
            'reg_weight': tune.loguniform(best_reg * 0.7, best_reg * 1.3),
            'learning_rate': tune.loguniform(best_lr * 0.85, best_lr * 1.15),
        }

def should_stop(iterations):
    """중단 조건"""
    if len(iterations) < 2:
        return False
    recalls = [it['recall@5'] for it in iterations]
    # 3회 연속 개선 없음 (더 여유있게)
    if len(recalls) >= 4 and recalls[-1] <= recalls[-2] and recalls[-2] <= recalls[-3] and recalls[-3] <= recalls[-4]:
        return True
    # 4회 연속 0.1% 미만 개선 (더 관대하게)
    if len(recalls) >= 5:
        improvements = [recalls[i] - recalls[i-1] for i in range(-3, 0)]
        if all(imp < 0.001 for imp in improvements):
            return True
    # 최대 8회 (증가)
    if len(iterations) >= 8:
        return True
    return False

def train_recbole(config_params):
    from ray import train
    config_dict = base_config.copy()
    config_dict.update({
        'model': MODEL_NAME,
        'embedding_size': int(config_params['embedding_size']),
        'n_layers': int(config_params['n_layers']),
        'reg_weight': config_params['reg_weight'],
        'learning_rate': config_params['learning_rate'],
    })
    try:
        config = Config(model=MODEL_NAME, config_dict=config_dict)
        init_seed(config['seed'], config['reproducibility'])
        dataset = create_dataset(config)
        train_data, valid_data, test_data = data_preparation(config, dataset)
        model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
        trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
        best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, verbose=False, show_progress=False)
        train.report({'recall@5': best_valid_result['recall@5'], 'ndcg@5': best_valid_result['ndcg@5'], 'recall@10': best_valid_result['recall@10']})
    except Exception as e:
        train.report({'recall@5': 0.0, 'ndcg@5': 0.0, 'recall@10': 0.0})

# Iterative tuning
print("=" * 80)
print("반복적 정밀 탐색 시작")
print("=" * 80)

iterations = []
scheduler = ASHAScheduler(metric='recall@5', mode='max', max_t=500, grace_period=10, reduction_factor=2)
search_alg = OptunaSearch(metric='recall@5', mode='max')
ray_results_path = str(Path('./ray_results').resolve())
resources_per_trial = {"cpu": 4, "gpu": 1.0} if device == 'cuda' else {"cpu": 2}
max_concurrent_trials = 1

for iteration in range(1, 9):  # 8 iterations (메모리 절약을 위해 더 작게 나눔)
    print(f"\n{'=' * 80}")
    print(f"ITERATION {iteration}")
    print(f"{'=' * 80}\n")

    search_space = get_search_space(iteration, iterations)
    num_trials = 8 if iteration == 1 else 6  # 메모리 절약: 20→8, 15→6

    print(f"탐색 공간: {search_space}")
    print(f"Trials: {num_trials}\n")

    tuner = tune.Tuner(
        tune.with_resources(train_recbole, resources=resources_per_trial),
        param_space=search_space,
        tune_config=tune.TuneConfig(scheduler=scheduler, search_alg=search_alg, num_samples=num_trials, max_concurrent_trials=max_concurrent_trials),
        run_config=RunConfig(name=f'lightgcn_refine_iter{iteration}', storage_path=ray_results_path),
    )

    start = time.time()
    results = tuner.fit()
    elapsed = time.time() - start

    best_result = results.get_best_result(metric='recall@5', mode='max')
    iterations.append({
        'iteration': iteration,
        'recall@5': best_result.metrics['recall@5'],
        'ndcg@5': best_result.metrics['ndcg@5'],
        'config': best_result.config,
        'improvement': best_result.metrics['recall@5'] - BASELINE_RECALL,
        'time': elapsed,
    })

    print(f"\n✅ Iteration {iteration} 완료 ({elapsed:.1f}초)")
    print(f"   Recall@5: {best_result.metrics['recall@5']:.4f} ({iterations[-1]['improvement']:+.4f} vs baseline)")
    print(f"   최적: emb={int(best_result.config['embedding_size'])}, layers={int(best_result.config['n_layers'])}, reg={best_result.config['reg_weight']:.6f}, lr={best_result.config['learning_rate']:.6f}")

    # 메모리 정리
    import gc
    del results, tuner, best_result
    gc.collect()

    if should_stop(iterations):
        print(f"\n🛑 중단: 개선 정체")
        break

# Best iteration
best_iter = max(iterations, key=lambda x: x['recall@5'])
print(f"\n{'=' * 80}")
print(f"최고 성능: Iteration {best_iter['iteration']}")
print(f"{'=' * 80}")
print(f"Recall@5: {best_iter['recall@5']:.4f} (baseline 대비 {best_iter['improvement']:+.4f})")
print(f"Config: {best_iter['config']}")

# Final training
print(f"\n{'=' * 80}")
print("최종 모델 학습")
print(f"{'=' * 80}")

final_config_dict = base_config.copy()
final_config_dict.update({
    'model': MODEL_NAME,
    'embedding_size': int(best_iter['config']['embedding_size']),
    'n_layers': int(best_iter['config']['n_layers']),
    'reg_weight': best_iter['config']['reg_weight'],
    'learning_rate': best_iter['config']['learning_rate'],
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

print(f"\n검증: Recall@5={best_valid_result['recall@5']:.4f}, NDCG@5={best_valid_result['ndcg@5']:.4f}")
print(f"테스트: Recall@5={test_result['recall@5']:.4f}, NDCG@5={test_result['ndcg@5']:.4f}")

# 추천 생성
all_users = dataset.inter_feat['user_id'].unique()
all_recommendations = {}
model.eval()
with torch.no_grad():
    for i, user_id in enumerate(all_users):
        user_external = dataset.id2token('user_id', user_id.item())
        topk_scores, topk_indices = full_sort_topk([user_id.item()], model, test_data, k=5, device=config['device'])
        items_external = [dataset.id2token('item_id', int(item)) for item in topk_indices[0].cpu().tolist()]
        all_recommendations[user_external] = items_external

# 제출 파일
result = [(user_id, ' '.join(recs)) for user_id, recs in all_recommendations.items()]
submission = pd.DataFrame(result, columns=['user_id', 'item_ids'])
t = pd.Timestamp.now()
output_dir = f"outputs/{t.year}-{t.month:02d}-{t.day:02d}"
os.makedirs(output_dir, exist_ok=True)
filename = f"{output_dir}/submit_LightGCN_Refinement_{t.year}{t.month:02d}{t.day:02d}{t.hour:02d}{t.minute:02d}{t.second:02d}.csv"
submission.to_csv(filename, index=False)

# 결과 저장
import json
result_file = f"{output_dir}/lightgcn_iterative_refinement_{t.year}{t.month:02d}{t.day:02d}{t.hour:02d}{t.minute:02d}{t.second:02d}.json"
with open(result_file, 'w') as f:
    json.dump({
        'best_iteration': best_iter['iteration'],
        'hyperparameters': {k: int(v) if k in ['embedding_size', 'n_layers'] else float(v) for k, v in best_iter['config'].items()},
        'validation_metrics': {'recall@5': float(best_valid_result['recall@5']), 'ndcg@5': float(best_valid_result['ndcg@5'])},
        'test_metrics': {'recall@5': float(test_result['recall@5']), 'ndcg@5': float(test_result['ndcg@5'])},
        'iterations_summary': [{'iteration': it['iteration'], 'recall@5': float(it['recall@5']), 'improvement': float(it['improvement'])} for it in iterations],
        'timestamp': t.strftime('%Y-%m-%d %H:%M:%S'),
    }, f, indent=2)

print(f"\n✅ 완료!")
print(f"제출: {filename}")
print(f"결과: {result_file}")

ray.shutdown()
