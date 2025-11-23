#!/usr/bin/env python3
"""
2개 모델만으로 앙상블 (MultiVAE + LightGCN)

EASE, RecVAE 제외 - 가장 균형잡힌 2개 모델만 사용
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict, Counter
from pathlib import Path

print("=" * 80)
print("2개 모델 앙상블 (MultiVAE + LightGCN)")
print("=" * 80)
print()

# 제출 파일 로드 (2개만)
submission_files = {
    'MultiVAE': 'outputs/2025-11-22/submit_MultiVAE_RayTune_20251122142530.csv',
    'LightGCN': 'outputs/2025-11-22/submit_LightGCN_RayTune_20251122142021.csv',
}

submissions = {}
for model_name, file_path in submission_files.items():
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        submissions[model_name] = df
        model_items = set()
        for items_str in df['item_ids']:
            model_items.update(items_str.split())
        print(f"✅ {model_name:10s}: {len(df):,}명, {len(model_items):,}개 아이템")
    else:
        print(f"⚠️  {model_name:10s}: 파일 없음")

print(f"\n✅ {len(submissions)}개 모델 로드 완료\n")

# 가중치 (테스트 성능 기반)
# MultiVAE: 0.0819, LightGCN: 0.0777 (5% 차이)
weights = {
    'MultiVAE': 0.55,   # 약간 더 높은 가중치
    'LightGCN': 0.45,
}

# Borda Count 설정
MAX_RANK = 20
TRUNCATE_AT = 20

print(f"모델별 가중치:")
for model, weight in weights.items():
    print(f"  {model:10s}: {weight:.2f}")
print()

# 전체 사용자 목록
all_users = set()
for df in submissions.values():
    all_users.update(df['user_id'].unique())
all_users = sorted(all_users)

print(f"총 사용자 수: {len(all_users):,}\n")

# Borda Count 앙상블
ensemble_recommendations = []

for i, user_id in enumerate(all_users):
    user_scores = defaultdict(float)

    for model_name, df in submissions.items():
        user_recs_str = df[df['user_id'] == user_id]['item_ids'].values
        if len(user_recs_str) > 0:
            user_recs = user_recs_str[0].split()
        else:
            user_recs = []

        if TRUNCATE_AT:
            user_recs = user_recs[:TRUNCATE_AT]

        model_weight = weights.get(model_name, 0.5)

        for rank, item_id in enumerate(user_recs, start=1):
            borda_score = model_weight * (MAX_RANK - rank + 1)
            if borda_score > 0:
                user_scores[item_id] += borda_score

    # Top 5
    top5_items = sorted(user_scores.items(), key=lambda x: -x[1])[:5]
    top5_item_ids = [item_id for item_id, score in top5_items]

    ensemble_recommendations.append({
        'user_id': user_id,
        'item_ids': ' '.join(top5_item_ids)
    })

    if (i + 1) % 1000 == 0:
        print(f"  진행: {i + 1:,}/{len(all_users):,} 사용자 처리 완료")

print(f"\n✅ 앙상블 완료: {len(ensemble_recommendations):,}개 추천 생성\n")

# 통계
ensemble_df = pd.DataFrame(ensemble_recommendations)

all_items = set()
for items_str in ensemble_df['item_ids']:
    all_items.update(items_str.split())

print("=" * 80)
print("앙상블 통계")
print("=" * 80)
print(f"총 사용자 수: {len(ensemble_df):,}")
print(f"고유 아이템 수: {len(all_items):,}")
print()

# 개별 모델과 비교
print(f"📊 아이템 커버리지 비교:")
for model_name, df in submissions.items():
    model_items = set()
    for items_str in df['item_ids']:
        model_items.update(items_str.split())
    print(f"  {model_name:10s}: {len(model_items):,}개")
print(f"  {'2-Model':10s}: {len(all_items):,}개")
print()

# Top 10
print(f"📊 가장 많이 추천된 아이템 (Top 10):")
item_counter = Counter()
for items_str in ensemble_df['item_ids']:
    item_counter.update(items_str.split())
top_items = item_counter.most_common(10)
for item_id, count in top_items:
    print(f"  {item_id}: {count}회")
print()

# 저장
t = pd.Timestamp.now()
output_dir = f"outputs/{t.year}-{t.month:02d}-{t.day:02d}"
os.makedirs(output_dir, exist_ok=True)
filename = f"{output_dir}/submit_Ensemble_2Models_{t.year}{t.month:02d}{t.day:02d}{t.hour:02d}{t.minute:02d}{t.second:02d}.csv"

ensemble_df.to_csv(filename, index=False)

print("=" * 80)
print("제출 파일 저장")
print("=" * 80)
print(f"✅ 파일명: {filename}")
print(f"   총 행 수: {len(ensemble_df):,}")
print(f"   고유 아이템 수: {len(all_items):,}")
print()

print("=" * 80)
print("✅ 완료!")
print("=" * 80)
