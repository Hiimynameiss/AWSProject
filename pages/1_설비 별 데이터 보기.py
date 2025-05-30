import streamlit as st
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
import os

# 사용 가능한 모듈만 지정
AVAILABLE_MODULES = [1, 2, 3, 4, 5, 11, 12, 13, 14, 15, 16, 17, 18]
MODULES = [os.path.join('csv', f'resampled_module{i}.csv') for i in AVAILABLE_MODULES]


@st.cache_data
def load_data(path):
    encodings = ['utf-8', 'utf-8-sig', 'cp949', 'euc-kr']
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"CSV 파일 {path}을 열 수 없습니다.")

def main():
    st.title("설비 별 데이터 셋 분석기")

    # 기본 정보 요약 (Text 형식)
    st.subheader("📊 기본 정보 요약")
    st.write(f"총 행 수: 33696013, 총 컬럼 수: 19")

# # 데이터 구조 요약 표 시각화
#     st.subheader("📝 데이터 구조 요약")
#     if not df.empty:
#         summary_data = []
#         for col in df.columns:
#             if col == 'localtime_dt':  # Skip generated datetime column
#                 continue
#             summary_data.append({
#                 "컬럼명": col,
#                 "데이터 타입": df[col].dtype,
#                 "고유값 수": df[col].nunique(),
#                 "결측값 수": df[col].isnull().sum()
#             })
#         summary_df = pd.DataFrame(summary_data)
#         st.dataframe(summary_df)
#     else:
#         st.warning("⚠️ 데이터프레임이 비어있습니다.")

    with st.sidebar:
        module_number = st.selectbox("모듈 선택", AVAILABLE_MODULES)
        min_allowed_date = datetime(2024, 12, 1)
        max_allowed_date = datetime(2025, 4, 30)
        start_date = st.date_input("시작 날짜", min_value=min_allowed_date, max_value=max_allowed_date, value=min_allowed_date)
        end_date = st.date_input("종료 날짜", min_value=min_allowed_date, max_value=max_allowed_date, value=max_allowed_date)
        analyze_button = st.button("분석하기")

    if analyze_button:
        DATA_PATH = os.path.join('csv', f'resampled_module{module_number}.csv')
        # [os.path.join('csv', f'resampled_module{i}.csv') for i in AVAILABLE_MODULES]

        df = load_data(DATA_PATH)

        # 날짜 변환
        df['localtime_dt'] = pd.to_datetime(df['localtime'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

        # 날짜 필터 설정 전 유효성 체크
        if df['localtime_dt'].isna().all():
            st.error("⚠️ 유효한 날짜 데이터가 없습니다. 'localtime' 컬럼을 확인해주세요.")
            return

        start_datetime = pd.to_datetime(start_date)
        end_datetime = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

        filtered_df = df[(df['localtime_dt'] >= start_datetime) & (df['localtime_dt'] <= end_datetime)]

        # 주요 지표 시각화
        st.subheader("📈 주요 지표 시각화 (시간별)")

        column_groups = {
            "Phase Voltages": ['voltageR', 'voltageS', 'voltageT'],
            "Line Voltages": ['voltageRS', 'voltageST', 'voltageTR'],
            "Currents": ['currentR', 'currentS', 'currentT'],
            "Power Factors": ['powerFactorR', 'powerFactorS', 'powerFactorT'],
            "Power Metrics": ['activePower', 'reactivePowerLagging']
        }

        for group_name, cols_in_group in column_groups.items():
            st.markdown(f"#### {group_name}")
            fig = go.Figure()

            for col_name in cols_in_group:
                fig.add_trace(go.Scatter(
                    x=filtered_df['localtime_dt'],
                    y=pd.to_numeric(filtered_df[col_name], errors='coerce'),
                    mode='lines',
                    name=col_name
                ))

            fig.update_layout(
                title=f"{group_name} 추이",
                xaxis_title="시간",
                yaxis_title="값",
                height=400,
                margin=dict(t=40, b=40),
                legend_title_text='지표'
            )
            st.plotly_chart(fig, use_container_width=True)

if __name__ == '__main__':
    main()
