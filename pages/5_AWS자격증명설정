import streamlit as st
import pandas as pd
import boto3
import json
from botocore.exceptions import ClientError, NoCredentialsError
import numpy as np

# AWS 클라이언트 설정 (환경변수 또는 AWS 설정을 통해 자격증명 필요)
@st.cache_resource
def get_sagemaker_client():
    try:
        return boto3.client('sagemaker-runtime', region_name='ap-northeast-2')
    except NoCredentialsError:
        st.error("❌ AWS 자격증명이 설정되지 않았습니다. AWS CLI 또는 환경변수를 설정해주세요.")
        return None

# 업로드
uploaded_files = st.file_uploader(
    "모듈별 테스트 파일 업로드 (module (1) ~ module (5), module (11) ~ module (18))", 
    type="csv", 
    accept_multiple_files=True
)

if uploaded_files:
    # 파일 이름 필터링 및 정렬 (module (1), module (2), ..., module (18))
    expected_modules = list(range(1, 6)) + list(range(11, 19))
    filtered_files = [f for f in uploaded_files if any(f.name == f"module ({i}).csv" for i in expected_modules)]
    sorted_files = sorted(filtered_files, key=lambda x: int(x.name.split("(")[1].split(")")[0]))

    if not sorted_files:
        st.warning("⚠️ 올바른 파일명의 CSV 파일이 없습니다. 파일명: 'module (1).csv', 'module (2).csv' 등")
        st.stop()

    df_list = []
    for file in sorted_files:
        try:
            df = pd.read_csv(file)
            # 데이터 유효성 검사
            if df.empty:
                st.warning(f"⚠️ {file.name}이 비어있습니다.")
                continue
            
            # NaN 값 처리
            df = df.fillna(0)  # 또는 적절한 대체값 사용
            
            df_list.append(df)
            st.success(f"✅ {file.name} 로드 완료 (행: {len(df)}, 열: {len(df.columns)})")
        except Exception as e:
            st.error(f"❌ {file.name} 읽기 실패: {e}")

    if df_list:
        df_combined = pd.concat(df_list, ignore_index=True)
        
        # 데이터 정보 표시
        st.write("### 📊 통합된 데이터 정보")
        st.write(f"**총 행 수:** {len(df_combined)}")
        st.write(f"**총 열 수:** {len(df_combined.columns)}")
        st.write("**컬럼명:**", list(df_combined.columns))
        
        # 데이터 미리보기
        with st.expander("데이터 미리보기"):
            st.dataframe(df_combined.head(10))
        
        # 데이터 타입 확인
        with st.expander("데이터 타입 정보"):
            st.write(df_combined.dtypes)

        # SageMaker 예측
        if st.button("📡 SageMaker 예측 요청"):
            sagemaker_client = get_sagemaker_client()
            
            if sagemaker_client is None:
                st.stop()
            
            endpoint_name = "tft-endpoint"
            
            try:
                # 데이터 전처리 및 검증
                st.info("🔄 데이터 전처리 중...")
                
                # 숫자가 아닌 데이터를 숫자로 변환 시도
                for col in df_combined.columns:
                    if df_combined[col].dtype == 'object':
                        df_combined[col] = pd.to_numeric(df_combined[col], errors='coerce')
                
                # NaN 값을 0으로 대체 (또는 다른 적절한 값)
                df_combined = df_combined.fillna(0)
                
                # 무한값 처리
                df_combined = df_combined.replace([np.inf, -np.inf], 0)
                
                # JSON 페이로드 준비
                payload = {
                    "instances": df_combined.to_dict(orient="records")
                }
                
                # 페이로드 크기 확인
                payload_size = len(json.dumps(payload).encode('utf-8'))
                st.info(f"📦 페이로드 크기: {payload_size / 1024:.2f} KB")
                
                if payload_size > 5 * 1024 * 1024:  # 5MB 제한
                    st.error("❌ 페이로드가 너무 큽니다. 데이터를 줄여주세요.")
                    st.stop()
                
                # SageMaker 엔드포인트 호출
                st.info("🚀 SageMaker 엔드포인트 호출 중...")
                
                response = sagemaker_client.invoke_endpoint(
                    EndpointName=endpoint_name,
                    ContentType='application/json',
                    Body=json.dumps(payload)
                )
                
                # 응답 처리
                result = json.loads(response['Body'].read().decode())
                
                st.success("🎉 예측 완료!")
                
                # 결과 표시
                with st.expander("예측 결과"):
                    st.json(result)
                
                # 결과를 DataFrame으로 변환 (가능한 경우)
                if isinstance(result, dict) and 'predictions' in result:
                    predictions_df = pd.DataFrame(result['predictions'])
                    st.write("### 📈 예측 결과 테이블")
                    st.dataframe(predictions_df)
                    
                    # CSV 다운로드 버튼
                    csv = predictions_df.to_csv(index=False)
                    st.download_button(
                        label="📥 예측 결과 다운로드 (CSV)",
                        data=csv,
                        file_name="prediction_results.csv",
                        mime="text/csv"
                    )
                
            except ClientError as e:
                error_code = e.response['Error']['Code']
                error_message = e.response['Error']['Message']
                
                if error_code == 'ValidationException':
                    st.error(f"❌ 입력 데이터 검증 실패: {error_message}")
                    st.info("💡 데이터 형식이나 스키마를 확인해주세요.")
                elif error_code == 'ModelError':
                    st.error(f"❌ 모델 오류: {error_message}")
                else:
                    st.error(f"❌ AWS 오류 ({error_code}): {error_message}")
                    
            except json.JSONDecodeError as e:
                st.error(f"❌ JSON 응답 파싱 실패: {e}")
                st.info("💡 엔드포인트가 올바른 JSON 형식을 반환하지 않습니다.")
                
            except Exception as e:
                st.error(f"🚨 예측 요청 실패: {str(e)}")
                st.info("💡 가능한 해결책:")
                st.info("1. AWS 자격증명 확인")
                st.info("2. 엔드포인트가 실행 중인지 확인")
                st.info("3. 입력 데이터 형식 확인")
                st.info("4. 네트워크 연결 확인")

    else:
        st.error("❌ 유효한 CSV 파일이 없습니다.")

# 사이드바에 도움말 추가
with st.sidebar:
    st.markdown("### 📋 사용 방법")
    st.markdown("""
    1. **AWS 자격증명 설정**
       - AWS CLI 구성 또는
       - 환경변수 설정
    
    2. **CSV 파일 업로드**
       - 파일명: module (1).csv ~ module (5).csv
       - 파일명: module (11).csv ~ module (18).csv
    
    3. **예측 실행**
       - 데이터 확인 후 예측 버튼 클릭
    """)
    
    st.markdown("### ⚠️ 주의사항")
    st.markdown("""
    - 엔드포인트가 실행 중이어야 함
    - 입력 데이터 형식이 모델과 일치해야 함
    - AWS 권한 설정 필요
    """)
