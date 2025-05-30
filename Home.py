import streamlit as st

st.set_page_config(page_title="Home", layout="wide")

st.title("🕵️‍♀️AWS Project 3팀")
st.header("❄️SMWU 시계열 데이터 분석 프로젝트")
# st.header("팀원 소개", anchor=None, help=None, divider='rainbow')

st.markdown("""
### 📂 프로젝트 페이지 안내

아래는 이 프로젝트의 주요 기능을 다루는 페이지들입니다. 사이드바에서 각 기능별 페이지로 이동해 상세한 분석 및 시각화를 확인하세요.

---

#### 📌 1. 설비 별 데이터 보기
- **기능**: 원하는 **설비**와 **날짜**를 선택하면 해당 조건에 맞는 **시간대별 주요 지표 시각화**를 확인할 수 있습니다.

#### ⚠️ 2. 운영 이상 감지 및 정제
- **기능 1**: 훈련 데이터에서 **설비 / 시간 범위 / 이상치 기준 에러값**을 선택해 **이상치 탐지 그래프**를 확인할 수 있습니다.
- **기능 2**: 정제된 CSV를 업로드하면, **이상치 제거 결과 시각화**도 함께 볼 수 있습니다.

#### ✅ 3. 검증용 기준 데이터 확인
- **기능**: 검증용 데이터인 `rtu_ground_truth_may.csv`에 대한 **기본 정보 요약, 컬럼 목록, 꺾은선 그래프 시각화**를 제공합니다.

#### 🔮 4. 에너지 예측 결과 보기
- **기능**: Amazon SageMaker의 엔드포인트로부터 예측값을 받아와, **activePower / 전기요금 / 탄소배출량** 등의 **시간대별 예측 결과를 시각화**할 수 있습니다.
""")



# # 각 팀원에 대한 정보
# with col1:
#     st.image("team/hayun.jpg", width=150)  # team 폴더 안에 있는 이미지 파일 경로 예시
#     st.subheader("안하연")
#     st.caption("전처리, 모델 학습, 엔드포인트 생성")

# with col2:
#     st.image("team/gain.jpg", width=150)
#     st.subheader("유가인")
#     st.caption("전처리, 모델 학습, 엔드포인트 생성")

# with col3:
#     st.image("team/seohyun.jpg", width=150)
#     st.subheader("송서현")
#     st.caption("streamlit")

# with col4:
#     st.image("team/seoyoung.jpg", width=150)
#     st.subheader("최서영")
#     st.caption("전처리, 모델 학습, 엔드포인트 생성")

