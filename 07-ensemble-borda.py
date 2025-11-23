#!/usr/bin/env python3
"""
RecBole Ensemble - Borda Count

4개 모델의 추천 결과를 Borda Count 알고리즘으로 앙상블:
- EASE: Recall@5 = 0.0718
- LightGCN: Recall@5 = 0.0849
- MultiVAE: Recall@5 = 0.0870
- RecVAE: Recall@5 = 0.0868

Borda Count 공식: score(item) = Σ weight × (MAX_RANK - rank + 1)
- MAX_RANK: 최대 순위 (기본값 20)
- rank: 모델별 아이템 순위 (1부터 시작)
- weight: 모델별 가중치

특징:
- RRF보다 선형적: 모든 순위에 공평한 가중치
- 직관적: 점수 기반 시스템 (스포츠 리그와 유사)
- 균형잡힘: 상위/하위 순위 모두 고려

예상 성능: RRF와 유사하거나 더 균형잡힌 추천 (다양성 증가)
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict, Counter
from pathlib import Path

print("=" * 60)
print("RecBole Ensemble - Borda Count")
print("=" * 60)
print()

# ============================================================
# 1. 제출 파일 로드
# ============================================================
print("=" * 60)
print("1. 제출 파일 로드")
print("=" * 60)

# 최신 제출 파일 경로 (2025-11-22 기준)
submission_files = {
    'EASE': 'outputs/2025-11-22/submit_EASE_RayTune_20251122141144.csv',
    'LightGCN': 'outputs/2025-11-22/submit_LightGCN_RayTune_20251122142021.csv',
    'MultiVAE': 'outputs/2025-11-22/submit_MultiVAE_RayTune_20251122142530.csv',
    'RecVAE': 'outputs/2025-11-22/submit_RecVAE_RayTune_20251122143202.csv',
}

# 데이터 로드
submissions = {}
for model_name, file_path in submission_files.items():
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        submissions[model_name] = df
        model_items = set()
        for items_str in df['item_ids']:
            model_items.update(items_str.split())
        print(f"✅ {model_name:10s}: {len(df):,}명 ({df['user_id'].nunique():,}명, {len(model_items):,}개 아이템)")
    else:
        print(f"⚠️  {model_name:10s}: 파일 없음 - {file_path}")

if len(submissions) < 2:
    print("\n❌ 에러: 최소 2개 이상의 제출 파일이 필요합니다")
    exit(1)

print(f"\n✅ 총 {len(submissions)}개 모델 로드 완료\n")

# ============================================================
# 2. Borda Count 앙상블 설정
# ============================================================
print("=" * 60)
print("2. Borda Count 앙상블 설정")
print("=" * 60)

# 모델별 가중치 (검증 Recall@5 기반 - 수정됨)
# MultiVAE > RecVAE > LightGCN > EASE
weights = {
    'MultiVAE': 0.35,  # 0.0870 (최고 성능)
    'RecVAE': 0.30,    # 0.0868
    'LightGCN': 0.25,  # 0.0849
    'EASE': 0.10       # 0.0718 (최저 성능)
}

# Borda Count 파라미터
MAX_RANK = 20      # 최대 순위 (상위 20개 고려)
TRUNCATE_AT = 20   # 각 모델의 상위 N개만 사용

print(f"MAX_RANK (최대 순위): {MAX_RANK}")
print(f"TRUNCATE_AT (절단 순위): {TRUNCATE_AT}")
print(f"\n모델별 가중치:")
for model in submissions.keys():
    weight = weights.get(model, 1.0 / len(submissions))
    print(f"  {model:10s}: {weight:.2f}")

print(f"\nBorda Score 공식:")
print(f"  score = weight × (MAX_RANK - rank + 1)")
print(f"  예시: 1위 = weight × 20, 10위 = weight × 11, 20위 = weight × 1")

print()

# ============================================================
# 3. Borda Count 앙상블 수행
# ============================================================
print("=" * 60)
print("3. Borda Count 앙상블 수행")
print("=" * 60)

# 전체 사용자 목록
all_users = set()
for df in submissions.values():
    all_users.update(df['user_id'].unique())
all_users = sorted(all_users)

print(f"총 사용자 수: {len(all_users):,}")

# 사용자별 앙상블 추천 생성
ensemble_recommendations = []

for i, user_id in enumerate(all_users):
    # 각 모델의 추천 결과를 Borda score로 변환
    user_scores = defaultdict(float)

    for model_name, df in submissions.items():
        # 해당 사용자의 추천 아이템 (공백으로 구분된 문자열)
        user_recs_str = df[df['user_id'] == user_id]['item_ids'].values
        if len(user_recs_str) > 0:
            user_recs = user_recs_str[0].split()  # 공백으로 분리
        else:
            user_recs = []

        # Truncation 적용
        if TRUNCATE_AT:
            user_recs = user_recs[:TRUNCATE_AT]

        # Borda 점수 계산
        model_weight = weights.get(model_name, 1.0 / len(submissions))

        for rank, item_id in enumerate(user_recs, start=1):
            # Borda formula: weight × (MAX_RANK - rank + 1)
            borda_score = model_weight * (MAX_RANK - rank + 1)

            # 양수 점수만 추가
            if borda_score > 0:
                user_scores[item_id] += borda_score

    # 점수 기준 정렬 후 상위 5개 선택
    top5_items = sorted(user_scores.items(), key=lambda x: -x[1])[:5]
    top5_item_ids = [item_id for item_id, score in top5_items]

    # 결과 추가 (사용자당 1행, 아이템들은 공백으로 구분)
    ensemble_recommendations.append({
        'user_id': user_id,
        'item_ids': ' '.join(top5_item_ids)
    })

    if (i + 1) % 1000 == 0:
        print(f"  진행: {i + 1:,}/{len(all_users):,} 사용자 처리 완료")

print(f"\n✅ 앙상블 완료: {len(ensemble_recommendations):,}개 추천 생성\n")

# ============================================================
# 4. 앙상블 통계
# ============================================================
print("=" * 60)
print("4. 앙상블 통계")
print("=" * 60)

ensemble_df = pd.DataFrame(ensemble_recommendations)

print(f"총 사용자 수: {len(ensemble_df):,}")
print(f"고유 사용자 수: {ensemble_df['user_id'].nunique():,}")

# 각 행의 item_ids를 공백으로 분리하여 전체 고유 아이템 수 계산
all_items = set()
for items_str in ensemble_df['item_ids']:
    all_items.update(items_str.split())
print(f"고유 아이템 수: {len(all_items):,}")
print(f"사용자당 평균 추천 수: {sum(len(items_str.split()) for items_str in ensemble_df['item_ids']) / len(ensemble_df):.2f}")

# 개별 모델과 아이템 커버리지 비교
print(f"\n📊 아이템 커버리지 비교:")
for model_name, df in submissions.items():
    model_items = set()
    for items_str in df['item_ids']:
        model_items.update(items_str.split())
    print(f"  {model_name:10s}: {len(model_items):,}개")
ensemble_coverage = len(all_items)
print(f"  {'BORDA':10s}: {ensemble_coverage:,}개")

# 가장 많이 추천된 아이템 Top 10
print(f"\n📊 가장 많이 추천된 아이템 (Top 10):")
item_counter = Counter()
for items_str in ensemble_df['item_ids']:
    item_counter.update(items_str.split())
top_items = item_counter.most_common(10)
for item_id, count in top_items:
    print(f"  {item_id}: {count}회")

print()

# ============================================================
# 5. 제출 파일 저장
# ============================================================
print("=" * 60)
print("5. 제출 파일 저장")
print("=" * 60)

t = pd.Timestamp.now()
output_dir = f"outputs/{t.year}-{t.month:02d}-{t.day:02d}"
os.makedirs(output_dir, exist_ok=True)
filename = f"{output_dir}/submit_Ensemble_BORDA_{t.year}{t.month:02d}{t.day:02d}{t.hour:02d}{t.minute:02d}{t.second:02d}.csv"

ensemble_df.to_csv(filename, index=False)

print(f"✅ 제출 파일 생성 완료!")
print(f"   파일명: {filename}")
print(f"   총 행 수: {len(ensemble_df):,}")
print(f"   고유 아이템 수: {len(all_items):,}")

# ============================================================
# 앙상블 결과 요약
# ============================================================
print()
print("=" * 60)
print("앙상블 결과 요약")
print("=" * 60)
print()

print(f"🤖 앙상블 정보:")
print(f"   방법: Borda Count")
print(f"   모델 수: {len(submissions)}개")
print(f"   사용 모델: {', '.join(submissions.keys())}")
print(f"   MAX_RANK: {MAX_RANK}")
print(f"   TRUNCATE_AT: {TRUNCATE_AT}")
print()

print(f"📊 성능 예상:")
print(f"   개별 모델 최고: Recall@5 = 0.0870 (MultiVAE)")
print(f"   Borda 예상: Recall@5 = 0.10-0.12 (균형잡힌 추천)")
print(f"   특징: RRF보다 다양성 높음, 모든 순위 공평 반영")
print()

print(f"💾 출력 파일:")
print(f"   {filename}")
print()

print("=" * 60)
print("✅ 모든 작업 완료!")
print("=" * 60)
