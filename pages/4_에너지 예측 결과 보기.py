import streamlit as st
import pandas as pd
import os
import json
# import requests # Not used in the provided snippet for SageMaker invocation
import boto3
from datetime import datetime, timedelta
import plotly.graph_objects as go
import numpy as np

# AWS SageMaker Runtime Client
# Ensure your AWS credentials and region are configured (e.g., via environment variables, ~/.aws/credentials)
try:
    #client = boto3.client("sagemaker-runtime", ) # Removed region_name to allow boto3 to determine it from config
    client = boto3.client("sagemaker-runtime", region_name="ap-northeast-2")  # 예: 서울 리전
except Exception as e:
    st.error(f"AWS Boto3 클라이언트 초기화 실패: {e}. AWS 설정(자격 증명, 리전)을 확인하세요.")
    st.stop()

st.title("🔍 전력 소비량 예측 및 분석")

# --- Configuration ---
SAGEMAKER_ENDPOINT_NAME = "tft-endpoint"  # S 실제 SageMaker 엔드포인트 이름으로 변경하세요!
ELECTRICITY_RATE_KWH = 180  # 원/kWh
CARBON_COEFFICIENT_KWH = 0.424  # kgCO2/kWh
TARGET_COLUMN = 'hourly_pow' # 예측 대상 컬럼 (검증용 데이터 셋의 일부.txt 기반 [cite: 2])
TIME_COLUMN = 'id'           # 시간 컬럼 (검증용 데이터 셋의 일부.txt 기반 [cite: 2])

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

# 업로드된 파일을 저장하는 함수 (기존 코드 유지)
def save_uploaded_file(directory, file):
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = os.path.join(directory, file.name)
    with open(file_path, 'wb') as f:
        f.write(file.getbuffer())
    return file_path

# ------------------------

# 🔍 이전 페이지에서 업로드한 파일 경로 사용 또는 직접 업로드
# st.session_state에서 file_path를 가져오거나, 이 페이지에서 직접 업로드하도록 수정
uploaded_file_path_from_session = st.session_state.get('uploaded_file_path', None)
uploaded_file_direct = st.file_uploader("예측을 위한 CSV 파일 업로드 (세션에 파일이 없는 경우)", type=['csv'])

# Google Drive 공유 링크를 통한 파일 불러오기
st.markdown("또는 👉 **Google Drive 공유 링크 입력**")
google_drive_url = st.text_input("🔗 Google Drive 공유 CSV 파일 URL", placeholder="https://drive.google.com/file/d/18r04ZNRd_Fz58Ay_g-7uY-6XK5q_P6V6/view?usp=sharing")

# 구글 드라이브 공유 링크를 다운로드 가능한 링크로 변환
def extract_google_drive_file_id(url):
    if "drive.google.com" in url:
        if "/file/d/" in url:
            return url.split("/file/d/")[1].split("/")[0]
        elif "id=" in url:
            return url.split("id=")[1]
    return None

def get_direct_download_url(file_id):
    return f"https://drive.google.com/uc?export=download&id={file_id}"

file_to_process = None
if google_drive_url:
    file_id = extract_google_drive_file_id(google_drive_url)
    if file_id:
        direct_url = get_direct_download_url(file_id)
        try:
            df_input_data = pd.read_csv(direct_url)
            st.success("✅ Google Drive 파일에서 데이터를 성공적으로 불러왔습니다.")
        except Exception as e:
            st.error(f"❌ Google Drive에서 파일을 불러오는 데 실패했습니다: {e}")
            st.stop()
    else:
        st.warning("⚠️ 유효한 Google Drive 공유 링크를 입력하세요.")


file_to_process = None
temp_file_path = None # 직접 업로드 시 임시 저장 경로

if uploaded_file_direct:
    # 사용자가 이 페이지에서 직접 파일을 업로드한 경우
    current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    direct_uploaded_filename = f"direct_upload_{current_time_str}_{uploaded_file_direct.name}"
    # save_uploaded_file 함수를 사용하기 위해 파일 객체의 name 속성을 설정
    class UploadedFileWithName:
        def __init__(self, file_obj, name):
            self._file_obj = file_obj
            self.name = name
        def getbuffer(self):
            return self._file_obj.getbuffer()

    file_to_process = UploadedFileWithName(uploaded_file_direct, direct_uploaded_filename)
    temp_file_path = save_uploaded_file("temp_uploads", file_to_process) # 임시 저장
    st.success(f"✅ 직접 업로드된 파일 사용: `{file_to_process.name}`")
elif uploaded_file_path_from_session:
    # 세션 상태에서 파일 경로를 사용하는 경우
    if os.path.exists(uploaded_file_path_from_session):
        file_to_process = uploaded_file_path_from_session
        st.success(f"✅ 세션에서 불러온 파일 사용: `{os.path.basename(file_to_process)}`")
    else:
        st.warning("세션의 파일 경로가 유효하지 않습니다. 파일을 직접 업로드해주세요.")
        st.stop()
else:
    st.info("📂 CSV 파일을 업로드하세요. (이전 페이지 또는 여기서 직접)")
    st.stop()

# 파일 처리 (경로 또는 파일 객체)
df_input_data = None
if file_to_process:
    try:
        # file_to_process가 경로 문자열인 경우와 파일 객체인 경우를 모두 처리
        file_path_for_reading = temp_file_path if temp_file_path else file_to_process

        encodings = ['utf-8', 'utf-8-sig', 'cp949', 'euc-kr']
        df_temp = None
        for enc in encodings:
            try:
                df_temp = pd.read_csv(file_path_for_reading, encoding=enc)
                break
            except UnicodeDecodeError:
                continue
        
        if df_temp is None:
            st.error("❌ CSV 파일을 열 수 없습니다. 인코딩 문제입니다.")
            st.stop()
        df_input_data = df_temp

    except Exception as e:
        st.error(f"❌ CSV 파일을 읽는 중 오류 발생: {e}")
        st.stop()
    finally:
        if temp_file_path and os.path.exists(temp_file_path): # 임시 파일 삭제
             try:
                os.remove(temp_file_path)
             except Exception:
                pass # 삭제 실패해도 계속 진행

if df_input_data is not None:
    # 데이터 전처리
    if TIME_COLUMN not in df_input_data.columns:
        st.error(f"'{TIME_COLUMN}' 컬럼이 파일에 없습니다. (시간 정보 컬럼)")
        st.stop()
    if TARGET_COLUMN not in df_input_data.columns:
        st.error(f"'{TARGET_COLUMN}' 컬럼이 파일에 없습니다. (예측 대상 컬럼)")
        st.stop()

    try:
        df_input_data[TIME_COLUMN] = pd.to_datetime(df_input_data[TIME_COLUMN])
        df_input_data[TARGET_COLUMN] = pd.to_numeric(df_input_data[TARGET_COLUMN], errors='coerce')
        df_input_data = df_input_data.dropna(subset=[TARGET_COLUMN]) # NaNs in target
        df_input_data = df_input_data.sort_values(by=TIME_COLUMN).reset_index(drop=True)
    except Exception as e:
        st.error(f"데이터 전처리 중 오류 발생 (시간 변환 또는 숫자 변환): {e}")
        st.stop()

    st.subheader("👀 입력 데이터 미리보기 (전처리 후)")
    st.dataframe(df_input_data.head())

    # 예측 기간(horizon) 설정
    # 검증용 데이터셋은 시간당 하나의 레코드를 가집니다. [cite: 2]
    min_horizon = 1
    max_horizon = min(168, len(df_input_data) - 1) # 최소 1개의 데이터는 모델 입력으로 남겨둠
    
    if max_horizon < min_horizon:
        st.warning(f"데이터가 너무 적어 예측을 수행할 수 없습니다. 최소 {min_horizon + 1}개의 데이터 포인트가 필요합니다.")
        st.stop()

    horizon = st.number_input("📈 예측 기간 (시간 단위, horizon)", min_value=min_horizon, max_value=max_horizon, value=min(6, max_horizon), step=1)

    # MAE 계산을 위한 데이터 분리
    actual_df_for_mae = df_input_data.iloc[-horizon:].copy()
    model_input_df = df_input_data.iloc[:-horizon].copy()

    if model_input_df.empty:
        st.error("모델에 입력할 데이터가 없습니다 (horizon 설정이 너무 큽니다). Horizon 값을 줄여주세요.")
        st.stop()

    st.markdown("---")
    st.subheader("🚀 추론 실행")
    
    # SageMaker 페이로드 구성 (모델의 기대 형식에 맞게 조정 필요)
    # 예시: 단순히 target series와 horizon을 전달
    # 실제 모델이 더 복잡한 입력을 요구할 수 있습니다 (e.g., start_time, features)
    payload = {
        # 다음은 일반적인 SageMaker 시계열 모델의 입력 형식 예시입니다.
        # 실제 사용 중인 모델의 입력 형식에 맞게 수정해야 합니다.
        "instances": [
            {
                "start": model_input_df[TIME_COLUMN].iloc[0].strftime("%Y-%m-%d %H:%M:%S"),
                "target": model_input_df[TARGET_COLUMN].tolist(),
                # "dynamic_feat": [[...]], # 만약 외부 특징을 사용한다면
            }
        ],
        "configuration": {
            "num_samples": 50, # 예측 샘플 수 (평균 예측만 필요하면 1 또는 제거)
            "output_types": ["mean"], # "mean", "quantiles", "samples" 등
            "quantiles": ["0.1", "0.5", "0.9"] # 만약 output_types에 "quantiles"가 있다면
        }
    }
    # 위의 페이로드는 DeepAR과 같은 SageMaker 내장 알고리즘의 일반적인 형태입니다.
    # 모델이 단순한 리스트나 다른 구조를 기대한다면 아래와 같이 수정:
    # payload = {
    #     "target_series": model_input_df[TARGET_COLUMN].tolist(),
    #     "horizon": horizon
    # }
    # 사용자의 모델에 맞게 payload를 정확히 구성하는 것이 중요합니다.

    if st.button("☁️ SageMaker로 추론 요청하기"):
        if SAGEMAKER_ENDPOINT_NAME == "tft-endpoint":
            st.error("SageMaker 엔드포인트 이름을 설정해주세요 (SAGEMAKER_ENDPOINT_NAME).")
        else:
            with st.spinner("SageMaker 엔드포인트에서 예측을 가져오는 중..."):
                sagemaker_result = invoke_sagemaker_endpoint(SAGEMAKER_ENDPOINT_NAME, payload)

            if sagemaker_result:
                st.success("✅ 추론 성공")
                st.session_state['sagemaker_result'] = sagemaker_result # 결과 저장
                st.session_state['actual_df_for_mae'] = actual_df_for_mae
                st.session_state['model_input_df'] = model_input_df
                st.session_state['horizon'] = horizon
                # 다음 섹션에서 결과를 표시하기 위해 페이지를 다시 로드할 필요는 없음
            else:
                st.error("❌ 추론 실패. 위의 오류 메시지를 확인하세요.")

    if 'sagemaker_result' in st.session_state:
        st.markdown("---")
        st.subheader("📊 추론 결과 분석")
        
        sagemaker_result = st.session_state['sagemaker_result']
        actual_df_for_mae = st.session_state['actual_df_for_mae']
        model_input_df = st.session_state['model_input_df']
        horizon = st.session_state['horizon']

        try:
            # SageMaker 결과 파싱 (모델의 응답 형식에 따라 크게 달라짐)
            # 예시: 응답이 {"predictions": [{"mean": [val1, val2, ...]}]} 형태일 경우
            if isinstance(sagemaker_result, dict) and "predictions" in sagemaker_result:
                if isinstance(sagemaker_result["predictions"], list) and \
                   len(sagemaker_result["predictions"]) > 0 and \
                   isinstance(sagemaker_result["predictions"][0], dict) and \
                   "mean" in sagemaker_result["predictions"][0] and \
                   len(sagemaker_result["predictions"][0]["mean"]) == horizon:
                    
                    predicted_values = sagemaker_result["predictions"][0]["mean"]
                else: # 단순 리스트 형태의 예측값으로 가정
                    predicted_values = sagemaker_result.get("predictions", []) # 또는 다른 키
                    if not isinstance(predicted_values, list) or len(predicted_values) != horizon :
                         st.error(f"SageMaker 응답에서 'predictions' 키를 찾을 수 없거나 예측 개수({len(predicted_values)})가 horizon({horizon})과 일치하지 않습니다.")
                         st.json(sagemaker_result) # 실제 응답 구조 확인용
                         st.stop()

            elif isinstance(sagemaker_result, list) and len(sagemaker_result) == horizon: # 응답이 예측값 리스트 자체일 경우
                predicted_values = sagemaker_result
            else: # 예측할 수 없는 구조
                st.error(f"SageMaker 응답의 구조를 해석할 수 없습니다. 'predictions' 키 또는 예측값 리스트를 확인하세요.")
                st.json(sagemaker_result) # 실제 응답 구조 확인용
                st.stop()

            # 예측값에 대한 타임스탬프 생성
            last_input_time = model_input_df[TIME_COLUMN].iloc[-1]
            # 검증용 데이터셋의 시간 간격은 1시간 [cite: 2]
            prediction_timestamps = pd.date_range(start=last_input_time + timedelta(hours=1), periods=horizon, freq='H')

            df_predictions = pd.DataFrame({
                TIME_COLUMN: prediction_timestamps,
                f'predicted_{TARGET_COLUMN}': predicted_values
            })

            # 1. ActivePower (hourly_pow) 예측 결과 시각화
            st.markdown(f"#### 1. {TARGET_COLUMN} 예측 결과")
            fig_power = go.Figure()
            fig_power.add_trace(go.Scatter(x=df_input_data[TIME_COLUMN], y=df_input_data[TARGET_COLUMN],
                                           mode='lines', name='실제값 (전체)', line=dict(color='blue')))
            fig_power.add_trace(go.Scatter(x=df_predictions[TIME_COLUMN], y=df_predictions[f'predicted_{TARGET_COLUMN}'],
                                           mode='lines+markers', name='예측값', line=dict(color='red')))
            # MAE 계산을 위해 사용된 실제값도 표시
            fig_power.add_trace(go.Scatter(x=actual_df_for_mae[TIME_COLUMN], y=actual_df_for_mae[TARGET_COLUMN],
                                           mode='markers', name='실제값 (MAE 비교 대상)', marker=dict(color='orange', size=8)))
            
            fig_power.update_layout(title=f'{TARGET_COLUMN} 실제값 vs. 예측값',
                                    xaxis_title='시간', yaxis_title='전력 소비량 (hourly_pow)')
            st.plotly_chart(fig_power, use_container_width=True)

            # 2. 전기 요금 및 탄소 배출량 예측 시각화
            st.markdown("#### 2. 예측된 전기 요금 및 탄소 배출량")
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
            
            st.dataframe(df_predictions[[TIME_COLUMN, f'predicted_{TARGET_COLUMN}', 'predicted_bill (원)', 'predicted_carbon (kgCO2)']].head())

            # 3. MAE 정확도 수치화 / 시각화
            st.markdown("#### 3. 모델 정확도 (MAE)")
            # 실제값과 예측값의 시간 정렬 및 병합
            comparison_df = pd.merge(actual_df_for_mae[[TIME_COLUMN, TARGET_COLUMN]],
                                     df_predictions[[TIME_COLUMN, f'predicted_{TARGET_COLUMN}']],
                                     on=TIME_COLUMN, how='inner')
            
            if comparison_df.empty or len(comparison_df) != horizon:
                st.warning(f"MAE를 계산하기 위한 실제값과 예측값의 매칭에 실패했습니다. (매칭된 행: {len(comparison_df)}, Horizon: {horizon}) 타임스탬프를 확인하세요.")
            else:
                mae = np.mean(np.abs(comparison_df[TARGET_COLUMN] - comparison_df[f'predicted_{TARGET_COLUMN}']))
                st.metric(label=f"평균 절대 오차 (MAE) for {TARGET_COLUMN}", value=f"{mae:.4f}")

                # MAE 시각화 (게이지 차트)
                max_mae_gauge = max(mae * 2, np.mean(actual_df_for_mae[TARGET_COLUMN]) * 0.5) # 게이지 최대값 동적 설정
                if max_mae_gauge == 0 : max_mae_gauge = 1 # 0으로 나누는 것 방지

                fig_mae_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=mae,
                    title={'text': f"MAE ({TARGET_COLUMN})<br><span style='font-size:0.8em;color:gray'>(작을수록 좋음)</span>"},
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
                fig_error.add_trace(go.Scatter(x=comparison_df[TIME_COLUMN], y=comparison_df[TARGET_COLUMN], mode='lines+markers', name='실제값'))
                fig_error.add_trace(go.Scatter(x=comparison_df[TIME_COLUMN], y=comparison_df[f'predicted_{TARGET_COLUMN}'], mode='lines+markers', name='예측값'))
                fig_error.update_layout(title='MAE 비교: 실제값 vs. 예측값 (예측 기간)', xaxis_title='시간', yaxis_title=TARGET_COLUMN)
                st.plotly_chart(fig_error, use_container_width=True)


        except Exception as e:
            st.error(f"결과 처리 또는 시각화 중 오류 발생: {e}")
            st.write("SageMaker 응답 데이터:")
            st.json(sagemaker_result) # 오류 발생 시 실제 응답 데이터 출력

# 세션 상태에 결과 저장 (다른 페이지로 넘길 경우)
# if 'sagemaker_result' in st.session_state and 'df_predictions' in locals():
# st.session_state["inference_results_for_viz"] = {
# "predictions_df": df_predictions.to_dict(),
# "mae": mae if 'mae' in locals() else None
# }
