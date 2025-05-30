import streamlit as st
import pandas as pd
import json
import boto3
from datetime import datetime, timedelta
import plotly.graph_objects as go
import numpy as np

# AWS SageMaker Runtime Client
try:
    client = boto3.client("sagemaker-runtime", region_name="ap-northeast-2")
except Exception as e:
    st.error(f"AWS Boto3 클라이언트 초기화 실패: {e}")
    st.stop()

st.title("전력 소비량 예측 및 분석")

# Configuration
SAGEMAKER_ENDPOINT_NAME = "tft-endpoint"
ELECTRICITY_RATE_KWH = 180  # 원/kWh
CARBON_COEFFICIENT_KWH = 0.424  # kgCO2/kWh
TARGET_COLUMN = 'activePower'
TIME_COLUMN = 'timestamp'

def invoke_sagemaker_endpoint(endpoint_name, payload_data):
    """SageMaker 엔드포인트 호출"""
    try:
        response = client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="application/json",
            Body=json.dumps(payload_data)
        )
        return json.loads(response["Body"].read().decode("utf-8"))
    except Exception as e:
        st.error(f"SageMaker 엔드포인트 호출 오류: {e}")
        return None

def convert_unix_timestamp_ms(timestamp):
    """Unix timestamp (밀리초)를 datetime으로 변환"""
    return pd.to_datetime(timestamp, unit='ms')

def extract_google_drive_file_id(url):
    """Google Drive URL에서 파일 ID 추출"""
    if "drive.google.com" in url:
        if "/file/d/" in url:
            return url.split("/file/d/")[1].split("/")[0]
        elif "id=" in url:
            return url.split("id=")[1]
    return None

def get_direct_download_url(file_id):
    """Google Drive 직접 다운로드 URL 생성"""
    return f"https://drive.google.com/uc?export=download&id={file_id}"

# Google Drive 링크 입력
st.write("Google Drive 공유 링크를 입력하세요")
google_drive_url = st.text_input(
    "CSV 파일 URL", 
    placeholder="https://drive.google.com/file/d/.../view?usp=sharing",
    value="https://drive.google.com/file/d/18r04ZNRd_Fz58Ay_g-7uY-6XK5q_P6V6/view?usp=sharing"
)

# 데이터 로딩
df_input_data = None

if google_drive_url:
    file_id = extract_google_drive_file_id(google_drive_url)
    if file_id:
        direct_url = get_direct_download_url(file_id)
        try:
            df_input_data = pd.read_csv(direct_url, header=0)
            df_input_data.columns = df_input_data.columns.str.strip()
            st.success("데이터를 성공적으로 불러왔습니다.")
        except Exception as e:
            st.error(f"파일을 불러오는데 실패했습니다: {e}")
    else:
        st.warning("유효한 Google Drive 공유 링크를 입력하세요.")

if df_input_data is not None:
    # 컬럼 확인
    if TIME_COLUMN not in df_input_data.columns or TARGET_COLUMN not in df_input_data.columns:
        st.error(f"필요한 컬럼이 없습니다. 사용 가능한 컬럼: {list(df_input_data.columns)}")
        st.stop()

    # 데이터 전처리
    try:
        df_input_data[TIME_COLUMN] = convert_unix_timestamp_ms(df_input_data[TIME_COLUMN])
        df_input_data[TARGET_COLUMN] = pd.to_numeric(df_input_data[TARGET_COLUMN], errors='coerce')
        df_input_data = df_input_data.dropna(subset=[TARGET_COLUMN])
        df_input_data = df_input_data.sort_values(by=TIME_COLUMN).reset_index(drop=True)
        st.success("데이터 전처리 완료")
    except Exception as e:
        st.error(f"데이터 전처리 오류: {e}")
        st.stop()

    # 데이터 미리보기
    st.subheader("데이터 미리보기")
    st.dataframe(df_input_data[[TIME_COLUMN, TARGET_COLUMN]].head(10))

    # 예측 기간 설정
    max_horizon = min(168, len(df_input_data) - 1)
    if max_horizon < 1:
        st.warning("데이터가 너무 적어 예측을 수행할 수 없습니다.")
        st.stop()

    horizon = st.number_input(
        "예측 기간 (시간 단위)", 
        min_value=1, 
        max_value=max_horizon, 
        value=min(6, max_horizon), 
        step=1
    )

    # 데이터 분리
    actual_df_for_mae = df_input_data.iloc[-horizon:].copy()
    model_input_df = df_input_data.iloc[:-horizon].copy()

    if model_input_df.empty:
        st.error("모델 입력 데이터가 없습니다. 예측 기간을 줄여주세요.")
        st.stop()

    # SageMaker 추론
    st.subheader("추론 실행")
    
    payload = {
        "instances": [{
            "start": model_input_df[TIME_COLUMN].iloc[0].strftime("%Y-%m-%d %H:%M:%S"),
            "target": model_input_df[TARGET_COLUMN].tolist(),
        }],
        "configuration": {
            "num_samples": 50,
            "output_types": ["mean"],
            "quantiles": ["0.1", "0.5", "0.9"]
        }
    }

    endpoint_name = st.text_input(
        "SageMaker 엔드포인트 이름", 
        value=SAGEMAKER_ENDPOINT_NAME,
        help="실제 배포된 SageMaker 엔드포인트 이름을 입력하세요"
    )

    if st.button("예측 실행"):
        if endpoint_name == "tft-endpoint":
            st.warning("실제 SageMaker 엔드포인트 이름을 입력해주세요.")
        else:
            with st.spinner("예측 중..."):
                sagemaker_result = invoke_sagemaker_endpoint(endpoint_name, payload)

            if sagemaker_result:
                st.success("예측 완료")
                
                # 결과 파싱
                predicted_values = None
                if isinstance(sagemaker_result, dict):
                    if "predictions" in sagemaker_result:
                        predictions = sagemaker_result["predictions"]
                        if isinstance(predictions, list) and len(predictions) > 0:
                            if isinstance(predictions[0], dict) and "mean" in predictions[0]:
                                predicted_values = predictions[0]["mean"]
                            else:
                                predicted_values = predictions
                elif isinstance(sagemaker_result, list):
                    predicted_values = sagemaker_result
                
                if predicted_values is None or len(predicted_values) != horizon:
                    st.error("예측 결과 파싱 실패")
                    st.json(sagemaker_result)
                    st.stop()

                # 예측 결과 처리
                last_input_time = model_input_df[TIME_COLUMN].iloc[-1]
                prediction_timestamps = pd.date_range(
                    start=last_input_time + timedelta(hours=1), 
                    periods=horizon, 
                    freq='H'
                )

                df_predictions = pd.DataFrame({
                    TIME_COLUMN: prediction_timestamps,
                    'predicted_activePower': predicted_values,
                    'predicted_bill': np.array(predicted_values) * ELECTRICITY_RATE_KWH,
                    'predicted_carbon': np.array(predicted_values) * CARBON_COEFFICIENT_KWH
                })

                # 날짜별 필터링 기능
                st.subheader("결과 분석")
                
                date_range = st.date_input(
                    "분석할 날짜 범위를 선택하세요",
                    value=(df_predictions[TIME_COLUMN].dt.date.min(), 
                           df_predictions[TIME_COLUMN].dt.date.max()),
                    min_value=df_predictions[TIME_COLUMN].dt.date.min(),
                    max_value=df_predictions[TIME_COLUMN].dt.date.max()
                )

                # 날짜 필터링
                if len(date_range) == 2:
                    start_date, end_date = date_range
                    mask = (df_predictions[TIME_COLUMN].dt.date >= start_date) & \
                           (df_predictions[TIME_COLUMN].dt.date <= end_date)
                    filtered_predictions = df_predictions[mask]
                else:
                    filtered_predictions = df_predictions

                if filtered_predictions.empty:
                    st.warning("선택한 날짜 범위에 데이터가 없습니다.")
                else:
                    # 시각화
                    st.subheader("예측된 전력 소비량")
                    fig1 = go.Figure()
                    fig1.add_trace(go.Scatter(
                        x=filtered_predictions[TIME_COLUMN], 
                        y=filtered_predictions['predicted_activePower'],
                        mode='lines+markers', 
                        name='예측 전력 소비량'
                    ))
                    fig1.update_layout(
                        title='시간대별 예측 전력 소비량',
                        xaxis_title='시간',
                        yaxis_title='전력 소비량 (kW)'
                    )
                    st.plotly_chart(fig1, use_container_width=True)

                    st.subheader("예측된 전기 요금")
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(
                        x=filtered_predictions[TIME_COLUMN], 
                        y=filtered_predictions['predicted_bill'],
                        mode='lines+markers', 
                        name='예측 전기 요금',
                        line=dict(color='green')
                    ))
                    fig2.update_layout(
                        title='시간대별 예측 전기 요금',
                        xaxis_title='시간',
                        yaxis_title='전기 요금 (원)'
                    )
                    st.plotly_chart(fig2, use_container_width=True)

                    st.subheader("예측된 탄소 배출량")
                    fig3 = go.Figure()
                    fig3.add_trace(go.Scatter(
                        x=filtered_predictions[TIME_COLUMN], 
                        y=filtered_predictions['predicted_carbon'],
                        mode='lines+markers', 
                        name='예측 탄소 배출량',
                        line=dict(color='red')
                    ))
                    fig3.update_layout(
                        title='시간대별 예측 탄소 배출량',
                        xaxis_title='시간',
                        yaxis_title='탄소 배출량 (kgCO2)'
                    )
                    st.plotly_chart(fig3, use_container_width=True)

                    # 요약 통계
                    st.subheader("예측 요약")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            "총 예측 전력 소비량", 
                            f"{filtered_predictions['predicted_activePower'].sum():.2f} kW"
                        )
                    with col2:
                        st.metric(
                            "총 예측 전기 요금", 
                            f"{filtered_predictions['predicted_bill'].sum():.0f} 원"
                        )
                    with col3:
                        st.metric(
                            "총 예측 탄소 배출량", 
                            f"{filtered_predictions['predicted_carbon'].sum():.2f} kgCO2"
                        )

                    # MAE 계산
                    st.subheader("모델 정확도 (MAE)")
                    comparison_df = pd.merge(
                        actual_df_for_mae[[TIME_COLUMN, TARGET_COLUMN]],
                        df_predictions[[TIME_COLUMN, 'predicted_activePower']],
                        on=TIME_COLUMN, 
                        how='inner'
                    )
                    
                    if not comparison_df.empty and len(comparison_df) == horizon:
                        mae = np.mean(np.abs(
                            comparison_df[TARGET_COLUMN] - 
                            comparison_df['predicted_activePower']
                        ))
                        st.metric("평균 절대 오차 (MAE)", f"{mae:.4f}")
                        
                        # MAE 비교 시각화
                        fig_mae = go.Figure()
                        fig_mae.add_trace(go.Scatter(
                            x=comparison_df[TIME_COLUMN], 
                            y=comparison_df[TARGET_COLUMN], 
                            mode='lines+markers', 
                            name='실제값'
                        ))
                        fig_mae.add_trace(go.Scatter(
                            x=comparison_df[TIME_COLUMN], 
                            y=comparison_df['predicted_activePower'], 
                            mode='lines+markers', 
                            name='예측값'
                        ))
                        fig_mae.update_layout(
                            title='실제값 vs 예측값 비교',
                            xaxis_title='시간',
                            yaxis_title='전력 소비량 (kW)'
                        )
                        st.plotly_chart(fig_mae, use_container_width=True)
                    else:
                        st.warning("MAE 계산을 위한 데이터 매칭에 실패했습니다.")

                    # 결과 데이터표
                    st.subheader("상세 예측 결과")
                    st.dataframe(filtered_predictions)

else:
    st.info("Google Drive 링크를 입력하여 CSV 파일을 불러오세요.")
