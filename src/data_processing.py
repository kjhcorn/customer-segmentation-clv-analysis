"""
데이터 처리 관련 함수들
Customer Segmentation & CLV Analysis Project
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


def load_data(file_path, file_type='auto'):
    """
    데이터 로딩 함수

    Parameters:
    file_path (str): 파일 경로
    file_type (str): 파일 타입 ('excel', 'csv', 'auto')

    Returns:
    pd.DataFrame: 로딩된 데이터
    """
    try:
        if file_type == 'auto':
            if file_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
            elif file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                raise ValueError("지원하지 않는 파일 형식입니다.")
        elif file_type == 'excel':
            df = pd.read_excel(file_path)
        elif file_type == 'csv':
            df = pd.read_csv(file_path)

        print(f"✅ 데이터 로딩 성공: {df.shape[0]:,} 행 × {df.shape[1]} 열")
        return df

    except Exception as e:
        print(f"❌ 데이터 로딩 실패: {e}")
        return None


def clean_data(df):
    """
    기본 데이터 정제

    Parameters:
    df (pd.DataFrame): 원본 데이터

    Returns:
    pd.DataFrame: 정제된 데이터
    """
    df_clean = df.copy()

    print("🧹 데이터 정제 시작...")

    # 1. 중복 제거
    initial_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    removed_duplicates = initial_rows - len(df_clean)
    if removed_duplicates > 0:
        print(f"   중복 제거: {removed_duplicates:,}건")

    # 2. CustomerID 결측값 제거 (RFM 분석 필수)
    if 'CustomerID' in df_clean.columns:
        initial_rows = len(df_clean)
        df_clean = df_clean.dropna(subset=['CustomerID'])
        removed_na = initial_rows - len(df_clean)
        if removed_na > 0:
            print(f"   CustomerID 결측값 제거: {removed_na:,}건")

    # 3. 날짜 형식 변환
    date_columns = ['InvoiceDate', 'FirstPurchaseDate']
    for col in date_columns:
        if col in df_clean.columns:
            df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
            print(f"   {col} 날짜 형식 변환 완료")

    print(f"✅ 데이터 정제 완료: {len(df_clean):,} 행")
    return df_clean


def handle_outliers(df, column, method='iqr', factor=1.5):
    """
    이상값 처리

    Parameters:
    df (pd.DataFrame): 데이터
    column (str): 대상 컬럼
    method (str): 방법 ('iqr', 'zscore')
    factor (float): IQR 배수 (기본 1.5)

    Returns:
    pd.Series: 이상값 마스크
    """
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR
        outliers = (df[column] < lower) | (df[column] > upper)

    elif method == 'zscore':
        from scipy import stats
        z_scores = np.abs(stats.zscore(df[column].dropna()))
        outliers = z_scores > 3

    outlier_count = outliers.sum()
    print(f"   {column} 이상값: {outlier_count:,}개 ({outlier_count / len(df) * 100:.1f}%)")

    return outliers
