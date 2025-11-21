#!/usr/bin/env python3
"""
RecBole Auto Ensemble - 자동 파일 탐지 및 가중치 계산

기능:
1. outputs/ 디렉토리에서 최신 제출 파일 자동 탐지
2. best_hyperparams JSON에서 검증 성능 자동 로드
3. 성능 기반 가중치 자동 계산
4. RRF 앙상블 실행

사용법:
    python 06-ensemble-auto.py                           # 모든 모델 자동 탐지
    python 06-ensemble-auto.py --models EASE LightGCN    # 특정 모델만
    python 06-ensemble-auto.py --date 2025-11-21         # 특정 날짜
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict
from pathlib import Path
import glob

def find_latest_submissions(output_dir='outputs', target_date=None, model_names=None):
    """
    최신 제출 파일 자동 탐지

    Args:
        output_dir: 출력 디렉토리
        target_date: 특정 날짜 (None이면 최신)
        model_names: 특정 모델 리스트 (None이면 전체)

    Returns:
        dict: {model_name: file_path}
    """
    submissions = {}

    # 날짜 디렉토리 목록
    if target_date:
        date_dirs = [os.path.join(output_dir, target_date)]
    else:
        date_dirs = sorted(glob.glob(os.path.join(output_dir, '2025-*')), reverse=True)

    # 각 모델별 최신 파일 찾기
    for date_dir in date_dirs:
        if not os.path.exists(date_dir):
            continue

        # submit_*.csv 파일 찾기
        submit_files = glob.glob(os.path.join(date_dir, 'submit_*_RayTune_*.csv'))

        for file_path in submit_files:
            filename = os.path.basename(file_path)
            # submit_ModelName_RayTune_timestamp.csv
            parts = filename.replace('.csv', '').split('_')
            if len(parts) >= 4:
                model_name = parts[1]  # EASE, LightGCN, MultiVAE, RecVAE

                # 모델 필터링
                if model_names and model_name not in model_names:
                    continue

                # 아직 없거나 더 최신 파일이면 업데이트
                if model_name not in submissions:
                    submissions[model_name] = file_path

    return submissions

def load_validation_scores(output_dir='outputs', model_name=None, target_date=None):
    """
    검증 성능 자동 로드

    Args:
        output_dir: 출력 디렉토리
        model_name: 모델 이름
        target_date: 특정 날짜

    Returns:
        float: recall@5 점수
    """
    # 날짜 디렉토리
    if target_date:
        date_dirs = [os.path.join(output_dir, target_date)]
    else:
        date_dirs = sorted(glob.glob(os.path.join(output_dir, '2025-*')), reverse=True)

    for date_dir in date_dirs:
        if not os.path.exists(date_dir):
            continue

        # best_hyperparams_{model}_{timestamp}.json 찾기
        pattern = f'best_hyperparams_{model_name.lower()}_*.json'
        json_files = glob.glob(os.path.join(date_dir, pattern))

        if json_files:
            # 가장 최신 파일
            latest_json = sorted(json_files, reverse=True)[0]

            try:
                with open(latest_json, 'r') as f:
                    data = json.load(f)
                    recall5 = data.get('validation_metrics', {}).get('recall@5', 0.0)
                    return recall5
            except:
                continue

    return None

def calculate_weights(scores):
    """
    검증 성능 기반 가중치 계산

    Args:
        scores: {model_name: recall@5}

    Returns:
        dict: {model_name: weight}
    """
    total_score = sum(scores.values())

    if total_score == 0:
        # 모든 점수가 0이면 균등 가중치
        return {name: 1.0 / len(scores) for name in scores.keys()}

    # 성능 비율로 가중치 계산
    weights = {name: score / total_score for name, score in scores.items()}
    return weights

def rrf_ensemble(submissions, weights, k=60):
    """
    RRF 앙상블

    Args:
        submissions: {model_name: DataFrame}
        weights: {model_name: weight}
        k: RRF constant

    Returns:
        DataFrame: 앙상블 결과
    """
    # 전체 사용자
    all_users = set()
    for df in submissions.values():
        all_users.update(df['resume_seq'].unique())
    all_users = sorted(all_users)

    ensemble_recommendations = []

    for user_id in all_users:
        user_scores = defaultdict(float)

        for model_name, df in submissions.items():
            user_recs = df[df['resume_seq'] == user_id]['recruitment_seq'].tolist()
            model_weight = weights.get(model_name, 1.0 / len(submissions))

            for rank, item_id in enumerate(user_recs, start=1):
                rrf_score = model_weight / (k + rank)
                user_scores[item_id] += rrf_score

        # 상위 5개
        top5_items = sorted(user_scores.items(), key=lambda x: -x[1])[:5]

        for item_id, score in top5_items:
            ensemble_recommendations.append({
                'resume_seq': user_id,
                'recruitment_seq': item_id
            })

    return pd.DataFrame(ensemble_recommendations)

def main():
    parser = argparse.ArgumentParser(description='RecBole Auto Ensemble')
    parser.add_argument('--models', nargs='+', help='특정 모델만 (예: EASE LightGCN)')
    parser.add_argument('--date', help='특정 날짜 (예: 2025-11-21)')
    parser.add_argument('--weights', nargs='+', type=float, help='수동 가중치 (모델 순서대로)')
    parser.add_argument('--k', type=int, default=60, help='RRF constant (기본값 60)')
    parser.add_argument('--output-dir', default='outputs', help='출력 디렉토리')

    args = parser.parse_args()

    print("=" * 60)
    print("RecBole Auto Ensemble")
    print("=" * 60)
    print()

    # 1. 제출 파일 자동 탐지
    print("=" * 60)
    print("1. 제출 파일 자동 탐지")
    print("=" * 60)

    submission_files = find_latest_submissions(
        output_dir=args.output_dir,
        target_date=args.date,
        model_names=args.models
    )

    if not submission_files:
        print("❌ 제출 파일을 찾을 수 없습니다")
        sys.exit(1)

    # 데이터 로드
    submissions = {}
    for model_name, file_path in submission_files.items():
        df = pd.read_csv(file_path)
        submissions[model_name] = df
        print(f"✅ {model_name:10s}: {file_path}")
        print(f"   └─ {len(df):,}개 추천 ({df['resume_seq'].nunique():,}명, "
              f"{df['recruitment_seq'].nunique():,}개 아이템)")

    print(f"\n✅ 총 {len(submissions)}개 모델 로드 완료\n")

    # 2. 가중치 계산
    print("=" * 60)
    print("2. 가중치 계산")
    print("=" * 60)

    if args.weights:
        # 수동 가중치
        if len(args.weights) != len(submissions):
            print(f"❌ 에러: 가중치 개수({len(args.weights)})와 모델 수({len(submissions)})가 다릅니다")
            sys.exit(1)

        weights = dict(zip(submissions.keys(), args.weights))
        print("수동 가중치:")
    else:
        # 자동 가중치 (검증 성능 기반)
        scores = {}
        for model_name in submissions.keys():
            recall5 = load_validation_scores(
                output_dir=args.output_dir,
                model_name=model_name,
                target_date=args.date
            )

            if recall5 is not None:
                scores[model_name] = recall5
                print(f"{model_name:10s}: Recall@5 = {recall5:.4f}")
            else:
                print(f"⚠️  {model_name:10s}: 검증 성능을 찾을 수 없음, 기본값 사용")
                scores[model_name] = 1.0 / len(submissions)

        weights = calculate_weights(scores)
        print("\n자동 계산된 가중치:")

    for model_name, weight in weights.items():
        print(f"  {model_name:10s}: {weight:.3f}")

    print(f"\nRRF constant (k): {args.k}\n")

    # 3. RRF 앙상블
    print("=" * 60)
    print("3. RRF 앙상블 수행")
    print("=" * 60)

    ensemble_df = rrf_ensemble(submissions, weights, k=args.k)

    print(f"✅ 앙상블 완료: {len(ensemble_df):,}개 추천 생성\n")

    # 4. 통계
    print("=" * 60)
    print("4. 앙상블 통계")
    print("=" * 60)

    print(f"총 추천 수: {len(ensemble_df):,}")
    print(f"고유 사용자: {ensemble_df['resume_seq'].nunique():,}")
    print(f"고유 아이템: {ensemble_df['recruitment_seq'].nunique():,}")
    print(f"사용자당 평균: {len(ensemble_df) / ensemble_df['resume_seq'].nunique():.2f}개\n")

    # 5. 저장
    print("=" * 60)
    print("5. 제출 파일 저장")
    print("=" * 60)

    t = pd.Timestamp.now()
    output_dir_date = f"{args.output_dir}/{t.year}-{t.month:02d}-{t.day:02d}"
    os.makedirs(output_dir_date, exist_ok=True)

    filename = f"{output_dir_date}/submit_Ensemble_RRF_Auto_{t.year}{t.month:02d}{t.day:02d}{t.hour:02d}{t.minute:02d}{t.second:02d}.csv"
    ensemble_df.to_csv(filename, index=False)

    print(f"✅ 제출 파일 생성 완료!")
    print(f"   파일명: {filename}")
    print(f"   총 행 수: {len(ensemble_df):,}")

    print("\n" + "=" * 60)
    print("✅ 모든 작업 완료!")
    print("=" * 60)

if __name__ == '__main__':
    main()
