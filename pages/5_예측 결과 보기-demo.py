import streamlit as st
import pandas as pd
import os
import json
import boto3
from datetime import datetime, timedelta
import plotly.graph_objects as go
import numpy as np

# AWS SageMaker Runtime Client
try:
    client = boto3.client("sagemaker-runtime", region_name="ap-northeast-2")  # 서울 리전
except Exception as e:
    st.error(f"AWS Boto3 클라이언트 초기화 실패: {e}. AWS 설정(자격 증명, 리전)을 확인하세요.")
    st.stop()

st.title("🔍 전력 소비량 예측 및 분석")

# --- Configuration ---
SAGEMAKER_ENDPOINT_NAME = "tft-endpoint"  # 실제 SageMaker 엔드포인트 이름으로 변경하세요!
ELECTRICITY_RATE_KWH = 180  # 원/kWh
CARBON_COEFFICIENT_KWH = 0.424  # kgCO2/kWh
TARGET_COLUMN = 'activePower' # 예측 대상 컬럼
TIME_COLUMN = 'timestamp'     # 시간 컬럼 (공백 제거)

# 추론 요청 함수
def invoke_sagemaker_endpoint(endpoint_name, payload_data):
    try:
        response = client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="application/json",
            Body=json.dumps(payload_data)
        )
        return json.loads(response["Body"].read().decode("utf-8"))
    except Exception as e:
        st.error(f"SageMaker 엔드포인트 호출 중 오류 발생: {e}")
        return None

# 업로드된 파일을 저장하는 함수
def save_uploaded_file(directory, file):
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = os.path.join(directory, file.name)
    with open(file_path, 'wb') as f:
        f.write(file.getbuffer())
    return file_path

# Unix timestamp (밀리초)를 datetime으로 변환하는 함수
def convert_unix_timestamp_ms(timestamp):
    """Unix timestamp (밀리초)를 pandas datetime으로 변환"""
    return pd.to_datetime(timestamp, unit='ms')

# 구글 드라이브 관련 함수들
def extract_google_drive_file_id(url):
    if "drive.google.com" in url:
        if "/file/d/" in url:
            return url.split("/file/d/")[1].split("/")[0]
        elif "id=" in url:
            return url.split("id=")[1]
    return None

def get_direct_download_url(file_id):
    return f"https://drive.google.com/uc?export=download&id={file_id}"

# 파일 업로드 및 Google Drive 링크 입력
uploaded_file_path_from_session = st.session_state.get('uploaded_file_path', None)
uploaded_file_direct = st.file_uploader("예측을 위한 CSV 파일 업로드", type=['csv'])

st.write("또는 👉 **Google Drive 공유 링크 입력**")
google_drive_url = st.text_input("🔗 Google Drive 공유 CSV 파일 URL", 
                                placeholder="https://drive.google.com/file/d/.../view?usp=sharing",
                                value="https://drive.google.com/file/d/18r04ZNRd_Fz58Ay_g-7uY-6XK5q_P6V6/view?usp=sharing")

# 데이터 로딩
df_input_data = None

# Google Drive에서 데이터 로딩
if google_drive_url:
    file_id = extract_google_drive_file_id(google_drive_url)
    if file_id:
        direct_url = get_direct_download_url(file_id)
        try:
            df_input_data = pd.read_csv(direct_url, header=0)
            df_input_data.columns = df_input_data.columns.str.strip()
            st.success("✅ Google Drive 파일에서 데이터를 성공적으로 불러왔습니다.")
        except Exception as e:
            st.error(f"❌ Google Drive에서 파일을 불러오는 데 실패했습니다: {e}")
            df_input_data = None
    else:
        st.warning("⚠️ 유효한 Google Drive 공유 링크를 입력하세요.")

# 직접 업로드된 파일 처리
elif uploaded_file_direct:
    try:
        df_input_data = pd.read_csv(uploaded_file_direct)
        df_input_data.columns = df_input_data.columns.str.strip()
        st.success(f"✅ 직접 업로드된 파일 사용: `{uploaded_file_direct.name}`")
    except Exception as e:
        st.error(f"❌ 업로드된 파일을 읽는 중 오류 발생: {e}")
        df_input_data = None

# 세션에서 파일 경로 사용
elif uploaded_file_path_from_session and os.path.exists(uploaded_file_path_from_session):
    try:
        encodings = ['utf-8', 'utf-8-sig', 'cp949', 'euc-kr']
        df_input_data = None
        for enc in encodings:
            try:
                df_input_data = pd.read_csv(uploaded_file_path_from_session, encoding=enc)
                df_input_data.columns = df_input_data.columns.str.strip()
                break
            except UnicodeDecodeError:
                continue
        
        if df_input_data is not None:
            st.success(f"✅ 세션에서 불러온 파일 사용: `{os.path.basename(uploaded_file_path_from_session)}`")
        else:
            st.error("❌ 세션 파일을 읽을 수 없습니다. 인코딩 문제입니다.")
    except Exception as e:
        st.error(f"❌ 세션 파일을 읽는 중 오류 발생: {e}")
        df_input_data = None

else:
    st.info("📂 CSV 파일을 업로드하거나 Google Drive 링크를 입력하세요.")

# 데이터가 로딩된 경우 처리 진행
if df_input_data is not None:
    # 컬럼 확인
    st.write("📋 데이터셋 컬럼:", list(df_input_data.columns))
    
    if TIME_COLUMN not in df_input_data.columns:
        st.error(f"'{TIME_COLUMN}' 컬럼이 파일에 없습니다. 사용 가능한 컬럼: {list(df_input_data.columns)}")
        st.stop()
    if TARGET_COLUMN not in df_input_data.columns:
        st.error(f"'{TARGET_COLUMN}' 컬럼이 파일에 없습니다. 사용 가능한 컬럼: {list(df_input_data.columns)}")
        st.stop()

    # 데이터 전처리
    try:
        # Unix timestamp (밀리초)를 datetime으로 변환
        df_input_data[TIME_COLUMN] = convert_unix_timestamp_ms(df_input_data[TIME_COLUMN])
        df_input_data[TARGET_COLUMN] = pd.to_numeric(df_input_data[TARGET_COLUMN], errors='coerce')
        df_input_data = df_input_data.dropna(subset=[TARGET_COLUMN])
        df_input_data = df_input_data.sort_values(by=TIME_COLUMN).reset_index(drop=True)
        
        st.success("✅ 데이터 전처리 완료")
    except Exception as e:
        st.error(f"❌ 데이터 전처리 중 오류 발생: {e}")
        st.write("타임스탬프 샘플 값:", df_input_data[TIME_COLUMN].head())
        st.stop()

    st.subheader("👀 입력 데이터 미리보기 (전처리 후)")
    st.dataframe(df_input_data[[TIME_COLUMN, TARGET_COLUMN]].head(10))

    # 예측 기간(horizon) 설정
    min_horizon = 1
    max_horizon = min(168, len(df_input_data) - 1)
    
    if max_horizon < min_horizon:
        st.warning(f"데이터가 너무 적어 예측을 수행할 수 없습니다. 최소 {min_horizon + 1}개의 데이터 포인트가 필요합니다.")
        st.stop()

    horizon = st.number_input("📈 예측 기간 (시간 단위, horizon)", 
                             min_value=min_horizon, 
                             max_value=max_horizon, 
                             value=min(6, max_horizon), 
                             step=1)

    # MAE 계산을 위한 데이터 분리
    actual_df_for_mae = df_input_data.iloc[-horizon:].copy()
    model_input_df = df_input_data.iloc[:-horizon].copy()

    if model_input_df.empty:
        st.error("모델에 입력할 데이터가 없습니다 (horizon 설정이 너무 큽니다). Horizon 값을 줄여주세요.")
        st.stop()

    st.write("---")
    st.subheader("🚀 추론 실행")
    
    # SageMaker 페이로드 구성
    payload = {
        "instances": [
            {
                "start": model_input_df[TIME_COLUMN].iloc[0].strftime("%Y-%m-%d %H:%M:%S"),
                "target": model_input_df[TARGET_COLUMN].tolist(),
            }
        ],
        "configuration": {
            "num_samples": 50,
            "output_types": ["mean"],
            "quantiles": ["0.1", "0.5", "0.9"]
        }
    }

    # 엔드포인트 이름 설정
    endpoint_name = st.text_input("🔗 SageMaker 엔드포인트 이름", 
                                 value=SAGEMAKER_ENDPOINT_NAME,
                                 help="실제 배포된 SageMaker 엔드포인트 이름을 입력하세요")

    if st.button("☁️ SageMaker로 추론 요청하기"):
        if endpoint_name == "tft-endpoint":
            st.warning("⚠️ 실제 SageMaker 엔드포인트 이름을 입력해주세요.")
        else:
            with st.spinner("SageMaker 엔드포인트에서 예측을 가져오는 중..."):
                sagemaker_result = invoke_sagemaker_endpoint(endpoint_name, payload)

            if sagemaker_result:
                st.success("✅ 추론 성공")
                st.session_state['sagemaker_result'] = sagemaker_result
                st.session_state['actual_df_for_mae'] = actual_df_for_mae
                st.session_state['model_input_df'] = model_input_df
                st.session_state['horizon'] = horizon
            else:
                st.error("❌ 추론 실패. 위의 오류 메시지를 확인하세요.")

    # 결과 분석 및 시각화
    if 'sagemaker_result' in st.session_state:
        st.write("---")
        st.subheader("📊 추론 결과 분석")
        
        sagemaker_result = st.session_state['sagemaker_result']
        actual_df_for_mae = st.session_state['actual_df_for_mae']
        model_input_df = st.session_state['model_input_df']
        horizon = st.session_state['horizon']

        try:
            # SageMaker 결과 파싱
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
            elif isinstance(sagemaker_result, list):
                predicted_values = sagemaker_result
            
            if predicted_values is None or len(predicted_values) != horizon:
                st.error(f"SageMaker 응답 파싱 실패. 예측값 개수: {len(predicted_values) if predicted_values else 0}, 필요 개수: {horizon}")
                st.json(sagemaker_result)
                st.stop()

            # 예측값에 대한 타임스탬프 생성
            last_input_time = model_input_df[TIME_COLUMN].iloc[-1]
            prediction_timestamps = pd.date_range(start=last_input_time + timedelta(hours=1), periods=horizon, freq='H')

            df_predictions = pd.DataFrame({
                TIME_COLUMN: prediction_timestamps,
                f'predicted_{TARGET_COLUMN}': predicted_values
            })

            # 1. ActivePower 예측 결과 시각화
            st.write(f"#### 1. {TARGET_COLUMN} 예측 결과")
            fig_power = go.Figure()
            fig_power.add_trace(go.Scatter(x=df_input_data[TIME_COLUMN], y=df_input_data[TARGET_COLUMN],
                                           mode='lines', name='실제값 (전체)', line=dict(color='blue')))
            fig_power.add_trace(go.Scatter(x=df_predictions[TIME_COLUMN], y=df_predictions[f'predicted_{TARGET_COLUMN}'],
                                           mode='lines+markers', name='예측값', line=dict(color='red')))
            fig_power.add_trace(go.Scatter(x=actual_df_for_mae[TIME_COLUMN], y=actual_df_for_mae[TARGET_COLUMN],
                                           mode='markers', name='실제값 (MAE 비교 대상)', marker=dict(color='orange', size=8)))
            
            fig_power.update_layout(title=f'{TARGET_COLUMN} 실제값 vs. 예측값',
                                    xaxis_title='시간', yaxis_title='전력 소비량 (kW)')
            st.plotly_chart(fig_power, use_container_width=True)

            # 2. 전기 요금 및 탄소 배출량 예측 시각화
            st.write("#### 2. 예측된 전기 요금 및 탄소 배출량")
            df_predictions['predicted_bill (원)'] = df_predictions[f'predicted_{TARGET_COLUMN}'] * ELECTRICITY_RATE_KWH
            df_predictions['predicted_carbon (kgCO2)'] = df_predictions[f'predicted_{TARGET_COLUMN}'] * CARBON_COEFFICIENT_KWH

            fig_derived = go.Figure()
            fig_derived.add_trace(go.Scatter(x=df_predictions[TIME_COLUMN], y=df_predictions['predicted_bill (원)'],
                                             mode='lines', name='예상 전기 요금 (원)', yaxis="y1"))
            fig_derived.add_trace(go.Scatter(x=df_predictions[TIME_COLUMN], y=df_predictions['predicted_carbon (kgCO2)'],
                                             mode='lines', name='예상 탄소 배출량 (kgCO2)', yaxis="y2"))

            fig_derived.update_layout(
                title='시간대별 예상 전기 요금 및 탄소 배출량',
                xaxis_title='시간',
                yaxis=dict(title='예상 전기 요금 (원)', side='left', showgrid=False),
                yaxis2=dict(title='예상 탄소 배출량 (kgCO2)', side='right', overlaying='y', showgrid=False),
                legend=dict(x=0.1, y=1.1, orientation="h")
            )
            st.plotly_chart(fig_derived, use_container_width=True)
            
            st.dataframe(df_predictions[[TIME_COLUMN, f'predicted_{TARGET_COLUMN}', 'predicted_bill (원)', 'predicted_carbon (kgCO2)']])

            # 3. MAE 정확도 수치화 / 시각화
            st.write("#### 3. 모델 정확도 (MAE)")
            comparison_df = pd.merge(actual_df_for_mae[[TIME_COLUMN, TARGET_COLUMN]],
                                     df_predictions[[TIME_COLUMN, f'predicted_{TARGET_COLUMN}']],
                                     on=TIME_COLUMN, how='inner')
            
            if comparison_df.empty or len(comparison_df) != horizon:
                st.warning(f"MAE를 계산하기 위한 실제값과 예측값의 매칭에 실패했습니다. (매칭된 행: {len(comparison_df)}, Horizon: {horizon})")
            else:
                mae = np.mean(np.abs(comparison_df[TARGET_COLUMN] - comparison_df[f'predicted_{TARGET_COLUMN}']))
                st.metric(label=f"평균 절대 오차 (MAE) for {TARGET_COLUMN}", value=f"{mae:.4f}")

                # MAE 시각화 (게이지 차트)
                max_mae_gauge = max(mae * 2, np.mean(actual_df_for_mae[TARGET_COLUMN]) * 0.5) if mae > 0 else 1

                fig_mae_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=mae,
                    title={'text': f"MAE ({TARGET_COLUMN}) - 작을수록 좋음"},
                    gauge={
                        'axis': {'range': [0, max_mae_gauge]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, max_mae_gauge * 0.25], 'color': "lightgreen"},
                            {'range': [max_mae_gauge * 0.25, max_mae_gauge * 0.5], 'color': "yellow"},
                            {'range': [max_mae_gauge * 0.5, max_mae_gauge], 'color': "lightcoral"}
                        ],
                    }
                ))
                fig_mae_gauge.update_layout(height=300)
                st.plotly_chart(fig_mae_gauge, use_container_width=True)
                
                # 오차 시각화 (실제 vs 예측)
                fig_error = go.Figure()
                fig_error.add_trace(go.Scatter(x=comparison_df[TIME_COLUMN], y=comparison_df[TARGET_COLUMN], 
                                              mode='lines+markers', name='실제값'))
                fig_error.add_trace(go.Scatter(x=comparison_df[TIME_COLUMN], y=comparison_df[f'predicted_{TARGET_COLUMN}'], 
                                              mode='lines+markers', name='예측값'))
                fig_error.update_layout(title='MAE 비교: 실제값 vs. 예측값 (예측 기간)', 
                                       xaxis_title='시간', yaxis_title=TARGET_COLUMN)
                st.plotly_chart(fig_error, use_container_width=True)

        except Exception as e:
            st.error(f"결과 처리 또는 시각화 중 오류 발생: {e}")
            st.write("SageMaker 응답 데이터:")
            st.json(sagemaker_result)
