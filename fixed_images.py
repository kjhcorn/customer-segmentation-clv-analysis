"""
수정된 포트폴리오 이미지 생성기
- 이모지 제거하고 텍스트로 대체
- numpy 호환성 문제 해결
"""

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

# 이미지 폴더 생성
os.makedirs('images', exist_ok=True)

# 고해상도 설정
plt.rcParams['savefig.dpi'] = 200
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['font.size'] = 12

print("🎨 수정된 포트폴리오 이미지 생성 시작")
print("=" * 50)


def create_project_overview():
    """프로젝트 개요 이미지 (이모지 제거)"""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    fig.patch.set_facecolor('white')

    # 메인 제목
    ax.text(5, 7.2, 'Customer Segmentation & CLV Analysis',
            fontsize=28, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))

    # 부제목
    ax.text(5, 6.5, 'RFM-T Analysis & ML-based Marketing Strategy',
            fontsize=18, ha='center', style='italic')

    # 핵심 성과 박스들 (이모지 제거)
    achievements = [
        ("797,815", "Transaction Data", "#FF6B6B"),
        ("5,939", "Customer Analysis", "#4ECDC4"),
        ("£600K+", "Annual Profit", "#45B7D1"),
        ("225%", "Expected ROI", "#FFA726")
    ]

    positions = [(2, 5), (4, 5), (6, 5), (8, 5)]

    for (number, desc, color), (x, y) in zip(achievements, positions):
        # 박스
        rect = plt.Rectangle((x - 0.8, y - 0.6), 1.6, 1.2,
                             facecolor=color, alpha=0.9,
                             edgecolor='black', linewidth=2)
        ax.add_patch(rect)

        # 숫자
        ax.text(x, y + 0.2, number, ha='center', va='center',
                fontsize=22, fontweight='bold', color='white')
        # 설명
        ax.text(x, y - 0.2, desc, ha='center', va='center',
                fontsize=12, fontweight='bold', color='white')

    # 기술 스택
    tech_stack = "Python • Pandas • Scikit-learn • Matplotlib • Seaborn"
    ax.text(5, 3.5, tech_stack, ha='center', va='center',
            fontsize=16, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgray', alpha=0.8))

    # 방법론
    methodology_text = """Core Analysis Methodology:
• RFM-T Analysis: Recency, Frequency, Monetary, Tenure
• CLV Modeling: Historic + Predictive + ML-based Approach  
• K-means Clustering: 8 features for Customer Segmentation
• Marketing Strategy: Customized campaigns with ROI optimization"""

    ax.text(5, 2.2, methodology_text, ha='center', va='center',
            fontsize=13,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9))

    # 하단 메시지
    ax.text(5, 0.5, 'Data-Driven Decision Making for Business Value Creation',
            ha='center', va='center', fontsize=16, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='gold', alpha=0.7))

    plt.tight_layout()
    plt.savefig('images/project_overview.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("✅ project_overview.png 생성 완료")


def create_rfm_distribution():
    """RFM-T 분포 시각화 (numpy 호환성 수정)"""
    np.random.seed(42)
    n_customers = 5939

    # 실제 프로젝트 결과와 유사한 분포 생성
    recency = np.random.gamma(2, 101, n_customers)  # 평균 202일
    frequency = np.random.gamma(1.2, 6.3, n_customers)  # 평균 7.6회
    monetary = np.random.gamma(1.1, 2400, n_customers)  # 평균 2650
    tenure = np.random.gamma(2.2, 217, n_customers)  # 평균 478일

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('RFM-T Variables Distribution Analysis', fontsize=20, fontweight='bold')

    # Recency
    axes[0, 0].hist(recency, bins=50, alpha=0.7, color='#FF6B6B', edgecolor='black', linewidth=0.5)
    axes[0, 0].set_title('[R] Recency Distribution\n(Days since last purchase)', fontsize=14)
    axes[0, 0].set_xlabel('Days')
    axes[0, 0].set_ylabel('Number of Customers')
    axes[0, 0].axvline(np.mean(recency), color='darkred', linestyle='--', linewidth=2,
                       label=f'Mean: {np.mean(recency):.0f} days')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # Frequency
    axes[0, 1].hist(frequency, bins=50, alpha=0.7, color='#4ECDC4', edgecolor='black', linewidth=0.5)
    axes[0, 1].set_title('[F] Frequency Distribution\n(Number of purchases)', fontsize=14)
    axes[0, 1].set_xlabel('Purchase Count')
    axes[0, 1].set_ylabel('Number of Customers')
    axes[0, 1].axvline(np.mean(frequency), color='darkgreen', linestyle='--', linewidth=2,
                       label=f'Mean: {np.mean(frequency):.1f} times')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # Monetary (95% 분위수까지) - 수정된 방법
    monetary_95 = np.percentile(monetary, 95)
    monetary_capped = monetary[monetary <= monetary_95]
    axes[1, 0].hist(monetary_capped, bins=50, alpha=0.7, color='#FFA726', edgecolor='black', linewidth=0.5)
    axes[1, 0].set_title('[M] Monetary Distribution\n(Total spend, 95th percentile)', fontsize=14)
    axes[1, 0].set_xlabel('Total Spend (£)')
    axes[1, 0].set_ylabel('Number of Customers')
    axes[1, 0].axvline(np.mean(monetary), color='darkorange', linestyle='--', linewidth=2,
                       label=f'Mean: £{np.mean(monetary):.0f}')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # Tenure
    axes[1, 1].hist(tenure, bins=50, alpha=0.7, color='#AB47BC', edgecolor='black', linewidth=0.5)
    axes[1, 1].set_title('[T] Tenure Distribution\n(Days since first purchase)', fontsize=14)
    axes[1, 1].set_xlabel('Days')
    axes[1, 1].set_ylabel('Number of Customers')
    axes[1, 1].axvline(np.mean(tenure), color='purple', linestyle='--', linewidth=2,
                       label=f'Mean: {np.mean(tenure):.0f} days')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('images/rfm_distribution.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("✅ rfm_distribution.png 생성 완료")


def create_customer_segments():
    """고객 세분화 결과"""
    segments = ['VIP Champions\n(Premium)', 'VIP Champions\n(General)', 'Standard\nCustomers', 'At Risk/\nLow Value']
    sizes = [11, 634, 4567, 727]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA726']
    avg_clvs = [195241, 8059, 2079, 617]

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('Customer Segmentation Results', fontsize=20, fontweight='bold')

    # 고객 분포 파이 차트
    wedges, texts, autotexts = axes[0].pie(sizes, labels=segments, autopct='%1.1f%%',
                                           colors=colors, startangle=90, textprops={'fontsize': 12})
    axes[0].set_title('Customer Distribution by Segment', fontsize=16, pad=20)

    # 평균 CLV 막대 차트
    bars = axes[1].bar(range(len(segments)), avg_clvs, color=colors, alpha=0.8)
    axes[1].set_title('Average Customer Lifetime Value by Segment', fontsize=16, pad=20)
    axes[1].set_ylabel('Average CLV (£)', fontsize=14)
    axes[1].set_xticks(range(len(segments)))
    axes[1].set_xticklabels(segments, fontsize=12, rotation=45, ha='right')

    # 값 표시
    for i, bar in enumerate(bars):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width() / 2., height + height * 0.01,
                     f'£{height:,}', ha='center', va='bottom', fontweight='bold')

    axes[1].grid(axis='y', alpha=0.3)
    axes[1].set_ylim(0, max(avg_clvs) * 1.1)

    plt.tight_layout()
    plt.savefig('images/customer_segments_overview.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("✅ customer_segments_overview.png 생성 완료")


def create_marketing_dashboard():
    """마케팅 전략 대시보드"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Marketing Strategy Dashboard', fontsize=20, fontweight='bold')

    segments = ['VIP Premium', 'VIP General', 'Standard', 'At Risk']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA726']
    budgets = [11000, 25000, 10000, 4000]
    roi_values = [400, 250, 175, 125]
    customer_counts = [11, 634, 4567, 727]

    # 1. 예산 배분 파이 차트
    wedges, texts, autotexts = axes[0, 0].pie(budgets, labels=segments, autopct='%1.1f%%',
                                              colors=colors, startangle=90)
    axes[0, 0].set_title('Monthly Marketing Budget Allocation\n(Total: £50,000)', fontsize=14)

    # 2. ROI 비교
    bars = axes[0, 1].bar(segments, roi_values, color=colors, alpha=0.8)
    axes[0, 1].set_title('Expected ROI by Segment', fontsize=14)
    axes[0, 1].set_ylabel('ROI (%)')
    axes[0, 1].tick_params(axis='x', rotation=45)

    for bar, roi in zip(bars, roi_values):
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width() / 2., height + 5,
                        f'{roi}%', ha='center', va='bottom', fontweight='bold')
    axes[0, 1].grid(axis='y', alpha=0.3)

    # 3. 투자 vs 수익
    returns = [b * (r / 100) for b, r in zip(budgets, roi_values)]

    x = np.arange(len(segments))
    width = 0.35

    bars1 = axes[1, 0].bar(x - width / 2, budgets, width, label='Investment',
                           color='lightcoral', alpha=0.8)
    bars2 = axes[1, 0].bar(x + width / 2, returns, width, label='Expected Return',
                           color='lightblue', alpha=0.8)

    axes[1, 0].set_title('Monthly Investment vs Expected Return', fontsize=14)
    axes[1, 0].set_ylabel('Amount (£)')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels([name.replace(' ', '\n') for name in segments])
    axes[1, 0].legend()
    axes[1, 0].grid(axis='y', alpha=0.3)

    # 4. 효율성 매트릭스
    bubble_sizes = [s / 10 for s in customer_counts]
    scatter = axes[1, 1].scatter(customer_counts, roi_values, s=bubble_sizes,
                                 c=colors, alpha=0.7, edgecolors='black', linewidth=2)
    axes[1, 1].set_title('Marketing Efficiency Matrix\n(Bubble size ∝ Customer count)', fontsize=14)
    axes[1, 1].set_xlabel('Number of Customers')
    axes[1, 1].set_ylabel('Expected ROI (%)')
    axes[1, 1].set_xscale('log')

    for i, (x, y, name) in enumerate(zip(customer_counts, roi_values, segments)):
        axes[1, 1].annotate(name, (x, y), xytext=(5, 5),
                            textcoords='offset points', fontsize=9)

    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('images/marketing_dashboard.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("✅ marketing_dashboard.png 생성 완료")


# 모든 이미지 생성 실행
if __name__ == "__main__":
    try:
        create_project_overview()
        create_rfm_distribution()
        create_customer_segments()
        create_marketing_dashboard()

        print(f"\n🎉 모든 포트폴리오 이미지 생성 완료!")
        print("=" * 50)
        print("생성된 이미지:")

        # 생성된 파일 확인
        image_files = ['project_overview.png', 'rfm_distribution.png',
                       'customer_segments_overview.png', 'marketing_dashboard.png']

        for img in image_files:
            img_path = f'images/{img}'
            if os.path.exists(img_path):
                size = os.path.getsize(img_path)
                print(f"✅ {img} - {size:,} bytes")
            else:
                print(f"❌ {img} - 생성 실패")

        print("=" * 50)

    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback

        traceback.print_exc()