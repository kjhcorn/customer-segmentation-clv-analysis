"""
시각화 함수들
Customer Segmentation & CLV Analysis Project
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# 기본 스타일 설정
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'


def plot_rfm_distribution(rfm_data, save_path=None):
    """
    RFM-T 분포 시각화

    Parameters:
    rfm_data (pd.DataFrame): RFM 데이터
    save_path (str): 저장 경로 (선택사항)
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('RFM-T Variables Distribution Analysis', fontsize=20, fontweight='bold')

    rfm_columns = ['Recency', 'Frequency', 'Monetary', 'Tenure']
    colors = ['#FF6B6B', '#4ECDC4', '#FFA726', '#AB47BC']
    icons = ['📅', '🛒', '💰', '⏱️']
    descriptions = [
        'Days since last purchase',
        'Number of purchases',
        'Total spend amount',
        'Days since first purchase'
    ]

    for i, (col, color, icon, desc) in enumerate(zip(rfm_columns, colors, icons, descriptions)):
        if col not in rfm_data.columns:
            continue

        row, col_idx = i // 2, i % 2
        ax = axes[row, col_idx]

        # 히스토그램
        if col == 'Monetary':
            # Monetary는 95% 분위수까지만 표시
            data_to_plot = rfm_data[col][rfm_data[col] <= rfm_data[col].quantile(0.95)]
        else:
            data_to_plot = rfm_data[col]

        ax.hist(data_to_plot, bins=50, alpha=0.7, color=color,
                edgecolor='black', linewidth=0.5)

        # 평균선
        mean_val = rfm_data[col].mean()
        ax.axvline(mean_val, color='darkred', linestyle='--', linewidth=2,
                   label=f'Mean: {mean_val:.0f}')

        # 제목 및 레이블
        ax.set_title(f'{icon} {col} Distribution\n({desc})', fontsize=14)
        ax.set_ylabel('Number of Customers')
        ax.legend()
        ax.grid(alpha=0.3)

        # 단위 설정
        if col in ['Recency', 'Tenure']:
            ax.set_xlabel('Days')
        elif col == 'Frequency':
            ax.set_xlabel('Count')
        elif col == 'Monetary':
            ax.set_xlabel('Amount (£)')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ RFM 분포 그래프 저장: {save_path}")

    plt.show()


def plot_customer_segments(segmentation_data, x_col='RFM_Score', y_col='Historic_CLV',
                           cluster_col='Cluster', segment_col='Segment_Name',
                           save_path=None):
    """
    고객 세분화 결과 시각화

    Parameters:
    segmentation_data (pd.DataFrame): 세분화 데이터
    x_col (str): X축 컬럼
    y_col (str): Y축 컬럼
    cluster_col (str): 클러스터 컬럼
    segment_col (str): 세그먼트 이름 컬럼
    save_path (str): 저장 경로
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Customer Segmentation Analysis', fontsize=20, fontweight='bold')

    # 색상 팔레트
    n_clusters = segmentation_data[cluster_col].nunique()
    colors = sns.color_palette('husl', n_clusters)

    # 1. 세그먼트별 산점도
    for i, cluster in enumerate(segmentation_data[cluster_col].unique()):
        cluster_data = segmentation_data[segmentation_data[cluster_col] == cluster]
        segment_name = cluster_data[segment_col].iloc[
            0] if segment_col in cluster_data.columns else f'Cluster {cluster}'

        axes[0, 0].scatter(cluster_data[x_col], cluster_data[y_col],
                           label=segment_name, alpha=0.6, s=50, c=[colors[i]])

    axes[0, 0].set_xlabel(x_col)
    axes[0, 0].set_ylabel(y_col)
    axes[0, 0].set_title(f'{x_col} vs {y_col} by Segment')
    axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. 세그먼트별 고객 수
    if segment_col in segmentation_data.columns:
        segment_counts = segmentation_data[segment_col].value_counts()
        wedges, texts, autotexts = axes[0, 1].pie(segment_counts.values,
                                                  labels=segment_counts.index,
                                                  autopct='%1.1f%%',
                                                  colors=colors[:len(segment_counts)])
        axes[0, 1].set_title('Customer Distribution by Segment')

    # 3. 세그먼트별 CLV 박스플롯
    if segment_col in segmentation_data.columns and y_col in segmentation_data.columns:
        box_data = []
        labels = []
        for segment in segmentation_data[segment_col].unique():
            segment_clv = segmentation_data[segmentation_data[segment_col] == segment][y_col]
            box_data.append(segment_clv)
            labels.append(segment.replace(' ', '\n'))  # 긴 이름 줄바꿈

        bp = axes[1, 0].boxplot(box_data, labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        axes[1, 0].set_title(f'{y_col} Distribution by Segment')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)

    # 4. Recency vs Frequency
    if all(col in segmentation_data.columns for col in ['Recency', 'Frequency']):
        for i, cluster in enumerate(segmentation_data[cluster_col].unique()):
            cluster_data = segmentation_data[segmentation_data[cluster_col] == cluster]
            segment_name = cluster_data[segment_col].iloc[
                0] if segment_col in cluster_data.columns else f'Cluster {cluster}'

            axes[1, 1].scatter(cluster_data['Recency'], cluster_data['Frequency'],
                               label=segment_name, alpha=0.6, s=50, c=[colors[i]])

        axes[1, 1].set_xlabel('Recency (days)')
        axes[1, 1].set_ylabel('Frequency (times)')
        axes[1, 1].set_title('Recency vs Frequency by Segment')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 세분화 그래프 저장: {save_path}")

    plt.show()


def plot_clv_analysis(clv_data, save_path=None):
    """
    CLV 분석 시각화

    Parameters:
    clv_data (pd.DataFrame): CLV 데이터
    save_path (str): 저장 경로
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Customer Lifetime Value (CLV) Analysis', fontsize=20, fontweight='bold')

    # 1. Historic CLV 분포
    if 'Historic_CLV' in clv_data.columns:
        historic_capped = clv_data['Historic_CLV'][
            clv_data['Historic_CLV'] <= clv_data['Historic_CLV'].quantile(0.95)
            ]
        axes[0, 0].hist(historic_capped, bins=50, alpha=0.7, color='#2E86AB',
                        edgecolor='black', linewidth=0.5)
        mean_clv = clv_data['Historic_CLV'].mean()
        axes[0, 0].axvline(mean_clv, color='darkblue', linestyle='--', linewidth=2,
                           label=f'Mean: £{mean_clv:,.0f}')
        axes[0, 0].set_title('Historic CLV Distribution\n(95th percentile)')
        axes[0, 0].set_xlabel('Historic CLV (£)')
        axes[0, 0].set_ylabel('Number of Customers')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)

    # 2. Predicted vs Historic CLV
    if all(col in clv_data.columns for col in ['Historic_CLV', 'Predicted_CLV']):
        sample_size = min(1000, len(clv_data))  # 샘플링으로 성능 향상
        sample_data = clv_data.sample(n=sample_size, random_state=42)

        axes[0, 1].scatter(sample_data['Historic_CLV'], sample_data['Predicted_CLV'],
                           alpha=0.6, s=30, color='#A23B72')

        # 완벽 예측선
        max_val = max(sample_data['Historic_CLV'].max(), sample_data['Predicted_CLV'].max())
        axes[0, 1].plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect Prediction')

        axes[0, 1].set_title('Predicted vs Historic CLV\n(ML Model Performance)')
        axes[0, 1].set_xlabel('Historic CLV (£)')
        axes[0, 1].set_ylabel('Predicted CLV (£)')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)

    # 3. AOV vs Purchase Count
    if all(col in clv_data.columns for col in ['AOV', 'Purchase_Count']):
        axes[1, 0].scatter(clv_data['AOV'], clv_data['Purchase_Count'],
                           alpha=0.6, s=30, color='#FF6B6B')
        axes[1, 0].set_title('Average Order Value vs Purchase Count')
        axes[1, 0].set_xlabel('Average Order Value (£)')
        axes[1, 0].set_ylabel('Purchase Count')
        axes[1, 0].grid(alpha=0.3)

    # 4. CLV 성장 전망
    months = ['Current', '3M', '6M', '12M']
    if 'Historic_CLV' in clv_data.columns:
        current_clv = clv_data['Historic_CLV'].mean()
        # 가정: 월 5% 성장
        growth_rate = 1.05
        projected_clv = [current_clv * (growth_rate ** (i * 3)) for i in range(4)]

        line = axes[1, 1].plot(months, projected_clv, marker='o', linewidth=3,
                               markersize=8, color='#2E86AB')
        axes[1, 1].fill_between(months, projected_clv, alpha=0.3, color='#2E86AB')
        axes[1, 1].set_title('Average CLV Growth Projection')
        axes[1, 1].set_ylabel('Average CLV (£)')
        axes[1, 1].grid(alpha=0.3)

        # 성장률 표시
        for i, val in enumerate(projected_clv[1:], 1):
            growth = ((val - projected_clv[0]) / projected_clv[0]) * 100
            axes[1, 1].annotate(f'+{growth:.0f}%',
                                xy=(i, val), xytext=(10, 10),
                                textcoords='offset points',
                                fontweight='bold', color='darkgreen')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ CLV 분석 그래프 저장: {save_path}")

    plt.show()