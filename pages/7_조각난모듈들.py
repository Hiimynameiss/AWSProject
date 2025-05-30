import streamlit as st
import pandas as pd
import requests
import json

# 업로드
uploaded_files = st.file_uploader("모듈별 테스트 파일 업로드 (module (1) ~ module (5), module (11) ~ module (18))", type="csv", accept_multiple_files=True)

if uploaded_files:
    # 파일 이름 필터링 및 정렬 (module (1), module (2), ..., module (18))
    filtered_files = [f for f in uploaded_files if any(f.name == f"module ({i}).csv" for i in list(range(1, 6)) + list(range(11, 19)))]
    sorted_files = sorted(filtered_files, key=lambda x: int(x.name.split("(")[1].split(")")[0]))

    df_list = []
    for file in sorted_files:
        try:
            df = pd.read_csv(file)
            df_list.append(df)
        except Exception as e:
            st.error(f"❌ {file.name} 읽기 실패: {e}")

    if df_list:
        df_combined = pd.concat(df_list, ignore_index=True)
        st.write("✅ 통합된 DataFrame:", df_combined.head())

        # 엔드포인트 전송
        if st.button("📡 SageMaker 예측 요청"):
            endpoint_url = "https://runtime.sagemaker.ap-northeast-2.amazonaws.com/endpoints/tft-endpoint/invocations"
            headers = {
                "Content-Type": "application/json",
                "X-Amz-Target": "SageMaker.InvokeEndpoint"
            }

            try:
                json_payload = json.dumps(df_combined.to_dict(orient="records"))
                response = requests.post(endpoint_url, data=json_payload, headers=headers)
                st.success("🎉 예측 완료!")
                st.json(response.json())
            except Exception as e:
                st.error(f"🚨 예측 요청 실패: {e}")
