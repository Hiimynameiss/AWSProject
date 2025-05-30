import streamlit as st
import pandas as pd
import os
from datetime import datetime
import plotly.graph_objects as go

def main():
    st.title("검증용 데이터 셋 정보 요약")

    # 👉 로컬 경로 설정
    file_path = r"C:\Users\Administrator\Desktop\SMWU\aws_project\rtu_ground_truth_may.csv"

    # 👉 인코딩 자동 처리
    encodings = ['utf-8', 'utf-8-sig', 'cp949', 'euc-kr']
    for enc in encodings:
        try:
            df = pd.read_csv(file_path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        st.error("❌ CSV 파일을 열 수 없습니다. 인코딩 문제입니다.")
        return

    # 👉 문자열 컬럼 변환
    df = df.astype({col: str for col in df.select_dtypes(include='object').columns})

    # 👉 기본 정보 요약
    st.subheader("📊 기본 정보 요약")
    col1, col2, col3 = st.columns(3)
    col1.metric("총 행 수", len(df))
    col2.metric("총 컬럼 수", len(df.columns))
    col3.metric("파일 크기", f"{os.path.getsize(file_path) / 1024:.2f} KB")

    # 👉 미리보기
    st.subheader("👀 데이터 미리보기 (상위 10개 행)")
    st.dataframe(df.head(10))

    # 👉 컬럼 목록
    st.subheader("🧾 컬럼 목록")
    st.markdown(", ".join([f"`{col}`" for col in df.columns]))

    # 👉 꺾은선 그래프 시각화
    st.subheader("📈 id 기준 주요 지표 꺾은선 그래프")
    numeric_cols = ['hourly_pow', 'may_bill', 'may_carbon', 'agg_pow']
    available_cols = [col for col in numeric_cols if col in df.columns]

    if 'id' in df.columns and available_cols:
        selected_col = st.selectbox("📌 시각화할 지표 선택", options=available_cols)

        if selected_col:
            df_plot = df[['id', selected_col]].copy()
            df_plot[selected_col] = pd.to_numeric(df_plot[selected_col], errors='coerce')
            df_plot = df_plot.dropna(subset=[selected_col])
            df_plot['id'] = df_plot['id'].astype(str)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_plot['id'],
                y=df_plot[selected_col],
                mode='lines+markers',
                name=selected_col
            ))

            fig.update_layout(
                title=f"{selected_col} (By id)",
                xaxis_title="id",
                yaxis_title=selected_col,
                height=300,
                margin=dict(t=40, b=40)
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("👈 하나 이상의 지표를 선택해주세요.")
    else:
        st.warning("⚠️ 'id' 또는 시각화 가능한 지표 컬럼이 누락되었습니다.")

if __name__ == '__main__':
    main()
