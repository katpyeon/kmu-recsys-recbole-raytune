#!/usr/bin/env python3
"""
LightGCN 실험 2: 한 번도 탐색하지 않은 파라미터

Baseline: Recall@5 = 0.0882
기존 탐색: embedding_size, n_layers, reg_weight, learning_rate

새로운 탐색:
1. train_batch_size: 배치 크기 영향 (항상 고정값 사용)
2. eval_batch_size: 평가 배치 크기
3. optimizer 타입: Adam vs SGD vs Adagrad
4. 극단적 embedding_size: 512, 384, 64 등

전략: 기존 최적값 고정 + 새 파라미터 탐색
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
print("LightGCN 미개척 파라미터 탐색")
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
    # 기존 최적값 고정
    'embedding_size': 256,
    'n_layers': 1,
    'reg_weight': 4e-5,
    'learning_rate': 0.004,
    'seed': 2024,
    'reproducibility': False,
    'show_progress': False,
    'worker': 4,
}

# 새로운 탐색 공간
search_space = {
    # 한 번도 튜닝 안 한 것들
    'train_batch_size': tune.choice([1024, 2048, 4096, 8192]),
    'eval_batch_size': tune.choice([4096, 51200, 102400]),

    # 극단적 embedding_size (기존: 128, 256만 시도)
    'embedding_size_extreme': tune.choice([64, 384, 512]),
}

print("탐색 공간:")
print(f"  기존 최적 고정: emb=256, layers=1, reg=4e-5, lr=0.004")
print(f"  🆕 train_batch_size: [1024, 2048, 4096, 8192]")
print(f"  🆕 eval_batch_size: [4096, 51200, 102400]")
print(f"  🆕 극단 embedding_size: [64, 384, 512]")
print(f"  Trials: 25\n")

def train_recbole(config_params):
    from ray import train
    config_dict = base_config.copy()

    # 새 파라미터 적용
    config_dict['train_batch_size'] = int(config_params['train_batch_size'])
    config_dict['eval_batch_size'] = int(config_params['eval_batch_size'])

    # 극단 embedding 선택 시 기존 값 오버라이드
    if config_params.get('embedding_size_extreme'):
        config_dict['embedding_size'] = int(config_params['embedding_size_extreme'])

    config_dict['model'] = MODEL_NAME

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
        print(f"❌ 실패: {e}")
        train.report({'recall@5': 0.0, 'ndcg@5': 0.0, 'recall@10': 0.0})

print("=" * 80)
print("최적화 시작")
print("=" * 80 + "\n")

scheduler = ASHAScheduler(metric='recall@5', mode='max', max_t=500, grace_period=10, reduction_factor=2)
search_alg = OptunaSearch(metric='recall@5', mode='max')
ray_results_path = str(Path('./ray_results').resolve())
resources_per_trial = {"cpu": 4, "gpu": 1.0} if device == 'cuda' else {"cpu": 2}
max_concurrent_trials = 1

tuner = tune.Tuner(
    tune.with_resources(train_recbole, resources=resources_per_trial),
    param_space=search_space,
    tune_config=tune.TuneConfig(scheduler=scheduler, search_alg=search_alg, num_samples=12, max_concurrent_trials=max_concurrent_trials),  # 메모리 절약: 25→12
    run_config=RunConfig(name='lightgcn_unexplored', storage_path=ray_results_path),
)

start = time.time()
results = tuner.fit()
elapsed = time.time() - start

print(f"\n✅ 최적화 완료 ({elapsed:.1f}초)\n")

# 결과 분석
best_result = results.get_best_result(metric='recall@5', mode='max')
best_config = best_result.config
best_metrics = best_result.metrics

print("=" * 80)
print("최적 결과")
print("=" * 80)
print(f"Recall@5: {best_metrics['recall@5']:.4f} (baseline 대비 {best_metrics['recall@5'] - BASELINE_RECALL:+.4f})")
print(f"\n최적 파라미터:")
print(f"  train_batch_size: {int(best_config['train_batch_size'])}")
print(f"  eval_batch_size: {int(best_config['eval_batch_size'])}")
print(f"  embedding_size: {int(best_config.get('embedding_size_extreme', 256))}")

# 파라미터별 성능 분석
results_df = results.get_dataframe()
successful_df = results_df[results_df['recall@5'] > 0]

print(f"\n📊 batch_size별 평균 성능:")
batch_perf = successful_df.groupby('config/train_batch_size')['recall@5'].agg(['mean', 'count'])
for bs, row in batch_perf.sort_values('mean', ascending=False).iterrows():
    print(f"  {int(bs):>5}: {row['mean']:.4f} (n={int(row['count'])})")

print(f"\n📊 embedding_size별 성능:")
emb_perf = successful_df.groupby('config/embedding_size_extreme')['recall@5'].agg(['mean', 'count'])
for emb, row in emb_perf.sort_values('mean', ascending=False).iterrows():
    print(f"  {int(emb):>3}: {row['mean']:.4f} (n={int(row['count'])})")

# 최종 학습
print(f"\n{'=' * 80}")
print("최종 모델 학습")
print(f"{'=' * 80}")

final_config_dict = base_config.copy()
final_config_dict.update({
    'model': MODEL_NAME,
    'train_batch_size': int(best_config['train_batch_size']),
    'eval_batch_size': int(best_config['eval_batch_size']),
    'embedding_size': int(best_config.get('embedding_size_extreme', 256)),
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
filename = f"{output_dir}/submit_LightGCN_Unexplored_{t.year}{t.month:02d}{t.day:02d}{t.hour:02d}{t.minute:02d}{t.second:02d}.csv"
submission.to_csv(filename, index=False)

# 결과 저장
import json
result_file = f"{output_dir}/lightgcn_unexplored_{t.year}{t.month:02d}{t.day:02d}{t.hour:02d}{t.minute:02d}{t.second:02d}.json"
with open(result_file, 'w') as f:
    json.dump({
        'hyperparameters': {
            'embedding_size': int(final_config_dict['embedding_size']),
            'n_layers': int(final_config_dict['n_layers']),
            'reg_weight': float(final_config_dict['reg_weight']),
            'learning_rate': float(final_config_dict['learning_rate']),
            'train_batch_size': int(final_config_dict['train_batch_size']),
            'eval_batch_size': int(final_config_dict['eval_batch_size']),
        },
        'validation_metrics': {'recall@5': float(best_valid_result['recall@5']), 'ndcg@5': float(best_valid_result['ndcg@5'])},
        'test_metrics': {'recall@5': float(test_result['recall@5']), 'ndcg@5': float(test_result['ndcg@5'])},
        'improvement_vs_baseline': float(best_metrics['recall@5'] - BASELINE_RECALL),
        'batch_size_analysis': {str(int(k)): {'mean': float(v['mean']), 'count': int(v['count'])} for k, v in batch_perf.iterrows()},
        'timestamp': t.strftime('%Y-%m-%d %H:%M:%S'),
    }, f, indent=2)

print(f"\n✅ 완료!")
print(f"제출: {filename}")
print(f"결과: {result_file}")

ray.shutdown()
