import streamlit as st
import pandas as pd
import boto3
import json
import io
from botocore.exceptions import ClientError, NoCredentialsError
import numpy as np
from datetime import datetime

# AWS 클라이언트 설정
@st.cache_resource
def get_sagemaker_client():
    try:
        return boto3.client('sagemaker-runtime', region_name='ap-northeast-2')
    except NoCredentialsError:
        st.error("❌ AWS 자격증명이 설정되지 않았습니다.")
        return None

def prepare_csv_for_sagemaker(df):
    """DataFrame을 SageMaker input_fn이 기대하는 CSV 문자열로 변환"""
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    return csv_buffer.getvalue()

def predict_batch(sagemaker_client, endpoint_name, df_batch, batch_id):
    """단일 배치에 대한 예측 수행"""
    try:
        # CSV 문자열로 변환 (SageMaker input_fn 형식에 맞춤)
        csv_data = prepare_csv_for_sagemaker(df_batch)
        
        st.info(f"🔄 배치 {batch_id} 예측 중... (크기: {len(csv_data.encode('utf-8'))/1024:.1f} KB)")
        
        response = sagemaker_client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='text/csv',  # input_fn의 content_type과 일치
            Body=csv_data
        )
        
        # 응답 처리
        result = json.loads(response['Body'].read().decode())
        
        # JSON 문자열을 DataFrame으로 변환
        if isinstance(result, str):
            result = json.loads(result)
        
        pred_df = pd.DataFrame(result)
        st.success(f"✅ 배치 {batch_id} 완료 (예측 결과: {len(pred_df)}행)")
        
        return pred_df
        
    except Exception as e:
        st.error(f"❌ 배치 {batch_id} 실패: {str(e)}")
        return None

# 메인 UI
st.title("🔮 TFT 모델 예측 시스템")

uploaded_files = st.file_uploader(
    "모듈별 테스트 파일 업로드 (module (1) ~ module (5), module (11) ~ module (18))", 
    type="csv", 
    accept_multiple_files=True
)

if uploaded_files:
    # 파일 필터링 및 정렬
    expected_modules = list(range(1, 6)) + list(range(11, 19))
    filtered_files = [f for f in uploaded_files if any(f.name == f"module ({i}).csv" for i in expected_modules)]
    sorted_files = sorted(filtered_files, key=lambda x: int(x.name.split("(")[1].split(")")[0]))

    if not sorted_files:
        st.warning("⚠️ 올바른 파일명의 CSV 파일이 없습니다.")
        st.stop()

    # 파일별 데이터 로드
    module_data = {}
    total_rows = 0
    
    for file in sorted_files:
        try:
            df = pd.read_csv(file)
            if df.empty:
                st.warning(f"⚠️ {file.name}이 비어있습니다.")
                continue
            
            # 모듈 번호 추출
            module_num = int(file.name.split("(")[1].split(")")[0])
            
            # 필수 컬럼 확인
            required_cols = ['localtime', 'activePower', 'voltageR', 'voltageS', 'voltageT',
                           'currentR', 'currentS', 'currentT', 'powerFactorR', 'powerFactorS', 
                           'powerFactorT', 'hour', 'dayofweek', 'month']
            
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                st.error(f"❌ {file.name}에 필수 컬럼이 없습니다: {missing_cols}")
                continue
            
            # module(equipment) 컬럼 추가 (SageMaker 코드에서 요구)
            df['module(equipment)'] = module_num
            
            # 데이터 전처리
            df = df.fillna(0)
            df = df.replace([np.inf, -np.inf], 0)
            
            module_data[module_num] = df
            total_rows += len(df)
            
            st.success(f"✅ {file.name} 로드 완료 (행: {len(df)})")
            
        except Exception as e:
            st.error(f"❌ {file.name} 읽기 실패: {e}")

    if module_data:
        # 전체 데이터 통합
        df_combined = pd.concat(module_data.values(), ignore_index=True)
        
        # 데이터 정보 표시
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 총 모듈 수", len(module_data))
        with col2:
            st.metric("📈 총 데이터 행", total_rows)
        with col3:
            estimated_size = len(df_combined.to_csv(index=False).encode('utf-8')) / 1024
            st.metric("💾 예상 크기", f"{estimated_size:.1f} KB")
        
        # 배치 크기 설정
        st.subheader("⚙️ 예측 설정")
        
        if estimated_size > 1000:  # 1MB 이상이면 배치 처리 권장
            st.warning("⚠️ 데이터가 큽니다. 배치 처리를 권장합니다.")
            default_batch = max(1, len(module_data) // 3)
        else:
            default_batch = len(module_data)
        
        batch_size = st.slider(
            "배치당 모듈 수 (작을수록 안전하지만 느림)", 
            min_value=1, 
            max_value=len(module_data),
            value=min(3, default_batch),
            help="한 번에 처리할 모듈 수를 설정합니다. 502 에러가 발생하면 이 값을 줄여보세요."
        )
        
        # 데이터 미리보기
        with st.expander("📋 데이터 미리보기"):
            st.dataframe(df_combined.head())
        
        # 모듈별 데이터 크기
        with st.expander("📊 모듈별 정보"):
            module_info = []
            for module_num, df in module_data.items():
                size_kb = len(df.to_csv(index=False).encode('utf-8')) / 1024
                module_info.append({
                    "모듈": f"module ({module_num})",
                    "행 수": len(df),
                    "크기 (KB)": f"{size_kb:.1f}"
                })
            st.dataframe(pd.DataFrame(module_info))

        # 예측 실행
        if st.button("🚀 SageMaker 예측 시작", type="primary"):
            sagemaker_client = get_sagemaker_client()
            
            if sagemaker_client is None:
                st.stop()
            
            endpoint_name = "tft-endpoint"
            
            try:
                # 모듈을 배치로 나누기
                module_nums = list(module_data.keys())
                batches = [module_nums[i:i+batch_size] for i in range(0, len(module_nums), batch_size)]
                
                st.info(f"📦 총 {len(batches)}개 배치로 나누어 처리합니다.")
                
                all_predictions = []
                progress_bar = st.progress(0)
                
                for batch_idx, batch_modules in enumerate(batches):
                    # 배치별 데이터 준비
                    batch_data = []
                    for module_num in batch_modules:
                        batch_data.append(module_data[module_num])
                    
                    df_batch = pd.concat(batch_data, ignore_index=True)
                    
                    # 예측 수행
                    pred_df = predict_batch(sagemaker_client, endpoint_name, df_batch, batch_idx + 1)
                    
                    if pred_df is not None:
                        all_predictions.append(pred_df)
                    
                    # 진행률 업데이트
                    progress_bar.progress((batch_idx + 1) / len(batches))
                
                # 결과 통합
                if all_predictions:
                    final_predictions = pd.concat(all_predictions, ignore_index=True)
                    
                    st.success(f"🎉 모든 예측 완료! 총 {len(final_predictions)}개 예측 결과")
                    
                    # 결과 표시
                    st.subheader("📈 예측 결과")
                    
                    # 모듈별 예측 개수
                    if 'module' in final_predictions.columns:
                        module_counts = final_predictions['module'].value_counts().sort_index()
                        st.write("**모듈별 예측 개수:**")
                        st.dataframe(module_counts.reset_index())
                    
                    # 전체 결과 미리보기
                    with st.expander("전체 예측 결과"):
                        st.dataframe(final_predictions)
                    
                    # CSV 다운로드
                    csv = final_predictions.to_csv(index=False)
                    st.download_button(
                        label="📥 예측 결과 다운로드 (CSV)",
                        data=csv,
                        file_name=f"tft_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
                else:
                    st.error("❌ 모든 배치가 실패했습니다.")
                    
            except Exception as e:
                st.error(f"🚨 예측 과정에서 오류 발생: {str(e)}")
                st.info("💡 해결 방법:")
                st.info("1. 배치 크기를 더 작게 설정")
                st.info("2. 엔드포인트 상태 확인")
                st.info("3. 데이터 형식 재확인")

# 사이드바 도움말
with st.sidebar:
    st.markdown("### 📋 사용 방법")
    st.markdown("""
    1. **CSV 파일 준비**
       - 파일명: module (1).csv ~ module (5).csv, module (11).csv ~ module (18).csv
       - 필수 컬럼: localtime, activePower, voltage/current/powerFactor (R,S,T), hour, dayofweek, month
    
    2. **배치 크기 조정**
       - 502 에러 발생 시 배치 크기를 1-2로 줄이기
       - 작은 배치는 안전하지만 느림
    
    3. **예측 실행**
       - 배치별로 순차 처리
       - 진행상황 실시간 확인
    """)
    
    st.markdown("### ⚠️ 문제 해결")
    st.markdown("""
    **502 Bad Gateway:**
    - 배치 크기를 1로 설정
    - 엔드포인트 상태 확인
    - 데이터 크기 확인
    
    **Timeout 오류:**
    - 더 작은 배치 사용
    - 데이터 양 줄이기
    """)
