import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import zipfile
import io
from detect_module5_light import get_anomaly_df as get_df5
from detect_module13_light import get_anomaly_df as get_df13
from detect_module15_light import get_anomaly_df as get_df15

st.set_page_config(page_title="운영 이상 감지 및 정제", layout="wide")
st.title("운영 이상 감지 및 정제 대시보드")

# 모듈 선택
module = st.selectbox("모듈 선택", ["module5", "module13", "module15"])
get_df = {"module5": get_df5, "module13": get_df13, "module15": get_df15}[module]
df = get_df()

# 타임 필터
df = df.sort_values("timestamp")
min_time = pd.to_datetime(df["timestamp"].min()).to_pydatetime()
max_time = pd.to_datetime(df["timestamp"].max()).to_pydatetime()
time_range = st.slider("⏱️ 시간 범위 선택", min_value=min_time, max_value=max_time,
                       value=(min_time, max_time), format="YYYY-MM-DD HH:mm")
df = df[(df["timestamp"] >= time_range[0]) & (df["timestamp"] <= time_range[1])]

# 이상치 기준 설정
threshold = st.slider("⚠️ 이상치 기준 에러값", min_value=0.0, max_value=2.0, value=1.0, step=0.1)
df["is_anomaly"] = df["total_error"] > threshold

# 이상치 수 요약
anomaly_count = df["is_anomaly"].sum()
st.markdown(f"### 🔍 이상치 감지 수: **:red[{anomaly_count}건]** / 총 {len(df)}건")

# 시각화
fig = go.Figure()

# 정상 라인
fig.add_trace(go.Scatter(
    x=df["timestamp"],
    y=df["total_error"],
    mode="lines",
    name="정상값",
    line=dict(color="steelblue", width=2)
))

# 이상치 포인트
fig.add_trace(go.Scatter(
    x=df[df["is_anomaly"]]["timestamp"],
    y=df[df["is_anomaly"]]["total_error"],
    mode="markers",
    name="이상치",
    marker=dict(color="red", size=10, symbol="circle-open-dot"),
))

fig.update_layout(
    title=f"{module.upper()} 전력 이상치 감지 결과",
    xaxis_title="시간",
    yaxis_title="오차 크기",
    height=500,
    font=dict(size=16),
    legend=dict(font=dict(size=14))
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("## 🧹 정제된 결과 확인")

# CSV 또는 ZIP 파일 업로드
uploaded_file = st.file_uploader("📂 정제된 파일 업로드 (CSV 또는 ZIP)", type=["csv", "zip"])

if uploaded_file is not None:
    try:
        df_cleaned = None
        
        # ZIP 파일 처리
        if uploaded_file.name.endswith('.zip'):
            with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
                csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]
                
                if not csv_files:
                    st.error("❌ ZIP 파일 내에 CSV 파일이 없습니다.")
                else:
                    # CSV 파일이 여러 개인 경우 선택하게 함
                    if len(csv_files) > 1:
                        selected_csv = st.selectbox("📋 처리할 CSV 파일 선택", csv_files)
                    else:
                        selected_csv = csv_files[0]
                    
                    st.info(f"📄 선택된 파일: {selected_csv}")
                    
                    # ZIP 내부 CSV 파일 읽기
                    with zip_ref.open(selected_csv) as csv_file:
                        df_cleaned = pd.read_csv(csv_file, parse_dates=["timestamp"])
        
        # 일반 CSV 파일 처리
        else:
            df_cleaned = pd.read_csv(uploaded_file, parse_dates=["timestamp"])
        
        # 데이터가 성공적으로 로드된 경우
        if df_cleaned is not None:
            df_cleaned = df_cleaned.sort_values("timestamp")

            MAX_ROWS = 5000
            df_small = df.copy()
            df_cleaned_small = df_cleaned.copy()

            if len(df_small) > MAX_ROWS:
                df_small = df_small.iloc[:MAX_ROWS]
            if len(df_cleaned_small) > MAX_ROWS:
                df_cleaned_small = df_cleaned_small.iloc[:MAX_ROWS]

            # 시각화할 컬럼 선택
            candidate_cols = ['activePower', 'currentR', 'currentS', 'currentT', 'powerFactorR', 'powerFactorS', 'powerFactorT']
            available_cols = [col for col in candidate_cols if col in df_cleaned.columns]

            if not available_cols:
                st.warning("⚠️ 정제된 데이터에 시각화 가능한 수치형 컬럼이 없습니다.")
            else:
                selected_col = st.selectbox("📊 정제 후 시각화할 컬럼 선택", available_cols)

                fig2 = go.Figure()

                if selected_col in df_small.columns:
                    y1 = pd.to_numeric(df_small[selected_col], errors='coerce')
                    st.write(f"정제 전 NaN 수: {y1.isna().sum()}")
                    fig2.add_trace(go.Scatter(
                        x=df_small["timestamp"], y=y1,
                        mode="lines", name="정제 전", line=dict(color="lightgray")
                    ))

                y2 = pd.to_numeric(df_cleaned_small[selected_col], errors='coerce')
                st.write(f"정제 후 NaN 수: {y2.isna().sum()}")

                fig2.add_trace(go.Scatter(
                    x=df_cleaned_small["timestamp"], y=y2,
                    mode="lines+markers", name="정제 후", line=dict(color="green", width=2)
                ))

                fig2.update_layout(
                    title=f"정제 전후 `{selected_col}` 비교",
                    xaxis_title="시간", yaxis_title=selected_col,
                    height=500
                )

                st.plotly_chart(fig2, use_container_width=True)

    except Exception as e:
        st.error(f"❌ 파일 처리 중 오류 발생: {e}")
