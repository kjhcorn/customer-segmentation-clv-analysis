"""
RFM-T 분석 계산 클래스
Customer Segmentation & CLV Analysis Project
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class RFMTAnalyzer:
    """RFM-T 분석을 위한 클래스"""

    def __init__(self, reference_date=None):
        """
        초기화

        Parameters:
        reference_date (datetime): 분석 기준일
        """
        if reference_date is None:
            self.reference_date = datetime.now()
        else:
            self.reference_date = reference_date
        print(f"📅 분석 기준일: {self.reference_date.strftime('%Y-%m-%d')}")

    def calculate_rfmt(self, df, customer_col='CustomerID',
                       date_col='InvoiceDate', revenue_col='Revenue',
                       invoice_col='InvoiceNo'):
        """
        RFM-T 변수 계산

        Parameters:
        df (pd.DataFrame): 거래 데이터
        customer_col (str): 고객 ID 컬럼
        date_col (str): 거래 날짜 컬럼
        revenue_col (str): 매출 컬럼
        invoice_col (str): 송장 번호 컬럼

        Returns:
        pd.DataFrame: RFM-T 계산 결과
        """
        print("🧮 RFM-T 변수 계산 중...")

        # 고객별 집계
        agg_dict = {
            date_col: ['min', 'max', 'count'],
            revenue_col: ['sum', 'mean'],
        }

        if invoice_col in df.columns:
            agg_dict[invoice_col] = 'nunique'

        rfmt_raw = df.groupby(customer_col).agg(agg_dict).round(2)

        # 컬럼명 정리
        if invoice_col in df.columns:
            rfmt_raw.columns = ['FirstPurchase', 'LastPurchase', 'TotalTransactions',
                                'TotalRevenue', 'AvgOrderValue', 'UniqueInvoices']
            rfmt_raw['Frequency'] = rfmt_raw['UniqueInvoices']
        else:
            rfmt_raw.columns = ['FirstPurchase', 'LastPurchase', 'TotalTransactions',
                                'TotalRevenue', 'AvgOrderValue']
            rfmt_raw['Frequency'] = rfmt_raw['TotalTransactions']

        # RFM-T 계산
        rfmt_raw['Recency'] = (self.reference_date - rfmt_raw['LastPurchase']).dt.days
        rfmt_raw['Monetary'] = rfmt_raw['TotalRevenue']
        rfmt_raw['Tenure'] = (self.reference_date - rfmt_raw['FirstPurchase']).dt.days

        print(f"✅ RFM-T 계산 완료: {len(rfmt_raw)}명 고객")

        return rfmt_raw.reset_index()

    def create_rfm_scores(self, rfmt_data, method='quantile'):
        """
        RFM-T 점수화 (1-5점)

        Parameters:
        rfmt_data (pd.DataFrame): RFM-T 원시 데이터
        method (str): 점수화 방법 ('quantile', 'equal_width')

        Returns:
        pd.DataFrame: 점수화된 데이터
        """
        print("📊 RFM-T 점수화 중...")

        scored_data = rfmt_data.copy()

        # 점수화할 컬럼들
        score_columns = ['Recency', 'Frequency', 'Monetary', 'Tenure']

        for col in score_columns:
            if col not in scored_data.columns:
                continue

            try:
                if col == 'Recency':
                    # Recency는 낮을수록 좋음 (최근 구매)
                    if method == 'quantile':
                        scored_data[f'{col}_Score'] = pd.qcut(
                            scored_data[col], q=5, labels=[5, 4, 3, 2, 1], duplicates='drop'
                        )
                    else:
                        scored_data[f'{col}_Score'] = pd.cut(
                            scored_data[col], bins=5, labels=[5, 4, 3, 2, 1]
                        )
                else:
                    # Frequency, Monetary, Tenure는 높을수록 좋음
                    if method == 'quantile':
                        scored_data[f'{col}_Score'] = pd.qcut(
                            scored_data[col], q=5, labels=[1, 2, 3, 4, 5], duplicates='drop'
                        )
                    else:
                        scored_data[f'{col}_Score'] = pd.cut(
                            scored_data[col], bins=5, labels=[1, 2, 3, 4, 5]
                        )

                print(f"   ✅ {col} 점수화 완료")

            except Exception as e:
                print(f"   ⚠️  {col} 점수화 실패: {e}")
                # 대체 방법: 단순 순위 기반
                if col == 'Recency':
                    scored_data[f'{col}_Score'] = 6 - pd.qcut(
                        scored_data[col].rank(), q=5, labels=False, duplicates='drop'
                    ) - 1
                else:
                    scored_data[f'{col}_Score'] = pd.qcut(
                        scored_data[col].rank(), q=5, labels=False, duplicates='drop'
                    ) + 1

        # 점수를 숫자형으로 변환
        score_cols = [f'{col}_Score' for col in score_columns if f'{col}_Score' in scored_data.columns]
        for col in score_cols:
            scored_data[col] = pd.to_numeric(scored_data[col], errors='coerce').fillna(3)

        # 종합 RFM 점수 계산
        if len(score_cols) >= 3:
            scored_data['RFM_Score'] = scored_data[score_cols].mean(axis=1).round(2)
            print(f"✅ 종합 RFM 점수 계산 완료")

        return scored_data