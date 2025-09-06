"""
CLV (Customer Lifetime Value) 모델링 클래스
Customer Segmentation & CLV Analysis Project
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import warnings

warnings.filterwarnings('ignore')


class CLVModel:
    """CLV 계산 및 예측을 위한 클래스"""

    def __init__(self, prediction_months=12):
        """
        초기화

        Parameters:
        prediction_months (int): CLV 예측 기간 (월)
        """
        self.prediction_months = prediction_months
        self.scaler = StandardScaler()
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.is_fitted = False
        self.feature_names = None

        print(f"💰 CLV 모델 초기화 (예측 기간: {prediction_months}개월)")

    def calculate_historic_clv(self, transaction_data, customer_col='CustomerID',
                               revenue_col='Revenue', date_col='InvoiceDate'):
        """
        Historic CLV 계산 (과거 실제 구매 금액 기반)

        Parameters:
        transaction_data (pd.DataFrame): 거래 데이터
        customer_col (str): 고객 ID 컬럼
        revenue_col (str): 매출 컬럼
        date_col (str): 날짜 컬럼

        Returns:
        pd.DataFrame: Historic CLV 결과
        """
        print("📊 Historic CLV 계산 중...")

        clv_data = transaction_data.groupby(customer_col).agg({
            revenue_col: ['sum', 'mean', 'count'],
            date_col: ['min', 'max']
        }).round(2)

        clv_data.columns = ['Historic_CLV', 'AOV', 'Purchase_Count',
                            'FirstPurchase', 'LastPurchase']

        # 고객 생애주기 계산
        clv_data['Lifespan_Days'] = (
                                            clv_data['LastPurchase'] - clv_data['FirstPurchase']
                                    ).dt.days + 1

        # 월평균 구매 빈도
        clv_data['Monthly_Frequency'] = clv_data['Purchase_Count'] / (
                clv_data['Lifespan_Days'] / 30.44
        )
        clv_data['Monthly_Frequency'] = clv_data['Monthly_Frequency'].fillna(
            clv_data['Purchase_Count']
        )  # 첫날 구매 고객 처리

        print(f"✅ Historic CLV 계산 완료: {len(clv_data)}명")
        print(f"   평균 Historic CLV: £{clv_data['Historic_CLV'].mean():.2f}")

        return clv_data.reset_index()

    def calculate_predictive_clv(self, clv_data):
        """
        Predictive CLV 계산 (단순 공식 기반)

        Parameters:
        clv_data (pd.DataFrame): Historic CLV 데이터

        Returns:
        pd.DataFrame: Predictive CLV 추가된 데이터
        """
        print(f"🔮 Predictive CLV 계산 중... (예측 기간: {self.prediction_months}개월)")

        data = clv_data.copy()

        # 예측 CLV = 월평균 구매빈도 × AOV × 예측개월수
        data['Predicted_CLV'] = (
                data['Monthly_Frequency'] *
                data['AOV'] *
                self.prediction_months
        ).round(2)

        # 극값 처리 (95% 분위수로 캡핑)
        cap_value = data['Predicted_CLV'].quantile(0.95)
        data['Predicted_CLV'] = np.minimum(data['Predicted_CLV'], cap_value)

        print(f"✅ Predictive CLV 계산 완료")
        print(f"   평균 Predicted CLV: £{data['Predicted_CLV'].mean():.2f}")

        return data

    def train_model(self, X, y):
        """
        ML 모델 학습

        Parameters:
        X (pd.DataFrame): 특성 데이터
        y (pd.Series): 타겟 (Historic CLV)

        Returns:
        dict: 모델 성능 지표
        """
        print("🤖 ML 모델 학습 중...")

        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 스케일링
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # 모델 학습
        self.model.fit(X_train_scaled, y_train)

        # 예측 및 평가
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_test = self.model.predict(X_test_scaled)

        # 성능 지표 계산
        performance = {
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test),
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test))
        }

        # 특성 중요도
        if hasattr(self.model, 'feature_importances_'):
            self.feature_names = X.columns.tolist()
            importance_dict = dict(zip(self.feature_names, self.model.feature_importances_))
            performance['feature_importance'] = sorted(
                importance_dict.items(), key=lambda x: x[1], reverse=True
            )

        self.is_fitted = True

        print(f"✅ 모델 학습 완료")
        print(f"   Test R²: {performance['test_r2']:.3f}")
        print(f"   Test MAE: £{performance['test_mae']:.2f}")

        return performance

    def predict_clv(self, X):
        """
        CLV 예측

        Parameters:
        X (pd.DataFrame): 특성 데이터

        Returns:
        np.array: 예측된 CLV
        """
        if not self.is_fitted:
            raise ValueError("모델이 학습되지 않았습니다. train_model()를 먼저 실행하세요.")

        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)

        return np.maximum(predictions, 0)  # 음수 제거