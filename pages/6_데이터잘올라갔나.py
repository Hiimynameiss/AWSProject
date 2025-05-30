import streamlit as st
import pandas as pd

st.title("📊 데이터 확인기")

# Google Drive 링크 입력
google_drive_url = st.text_input("🔗 Google Drive CSV 파일 URL을 입력하세요:", 
                                placeholder="https://drive.google.com/file/d/...")

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

# 파일 업로드 옵션도 추가
st.markdown("**또는**")
uploaded_file = st.file_uploader("📂 CSV 파일을 직접 업로드하세요:", type=['csv'])

df = None

# Google Drive에서 파일 로드
if google_drive_url:
    file_id = extract_google_drive_file_id(google_drive_url)
    if file_id:
        direct_url = get_direct_download_url(file_id)
        try:
            df = pd.read_csv(direct_url)
            st.success("✅ Google Drive에서 파일을 성공적으로 불러왔습니다!")
        except Exception as e:
            st.error(f"❌ 파일 로드 실패: {e}")
    else:
        if google_drive_url.strip():  # 빈 문자열이 아닌 경우에만 경고
            st.warning("⚠️ 유효한 Google Drive 링크를 입력하세요.")

# 직접 업로드된 파일 로드
elif uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("✅ 파일을 성공적으로 업로드했습니다!")
    except Exception as e:
        st.error(f"❌ 파일 로드 실패: {e}")

# 데이터가 로드되면 분석 시작
if df is not None:
    st.markdown("---")
    st.subheader("📋 데이터 정보")
    
    # 기본 정보
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("행 개수", len(df))
    with col2:
        st.metric("열 개수", len(df.columns))
    with col3:
        st.metric("메모리 사용량", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
    
    # 컬럼 정보
    st.subheader("🔍 컬럼 정보")
    col_info = pd.DataFrame({
        '컬럼명': df.columns,
        '데이터 타입': df.dtypes.astype(str),
        '결측값 개수': df.isnull().sum(),
        '결측값 비율(%)': (df.isnull().sum() / len(df) * 100).round(2),
        '고유값 개수': df.nunique()
    })
    st.dataframe(col_info)
    
    # 데이터 미리보기
    st.subheader("👀 데이터 미리보기")
    st.write("**처음 5개 행:**")
    st.dataframe(df.head())
    
    st.write("**마지막 5개 행:**")
    st.dataframe(df.tail())
    
    # 숫자형 컬럼 통계
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        st.subheader("📊 숫자형 컬럼 통계")
        st.dataframe(df[numeric_cols].describe())
    
    # 문자형 컬럼 정보
    text_cols = df.select_dtypes(include=['object']).columns
    if len(text_cols) > 0:
        st.subheader("📝 문자형 컬럼 정보")
        for col in text_cols:
            with st.expander(f"'{col}' 컬럼 세부 정보"):
                st.write(f"고유값 개수: {df[col].nunique()}")
                st.write("상위 5개 값:")
                st.write(df[col].value_counts().head())
    
    # 데이터 타입 변환 제안
    st.subheader("💡 데이터 타입 변환 제안")
    suggestions = []
    
    for col in df.columns:
        if df[col].dtype == 'object':
            # 날짜/시간으로 변환 가능한지 체크
            if any(keyword in col.lower() for keyword in ['time', 'date', '시간', '날짜', 'timestamp']):
                try:
                    pd.to_datetime(df[col].head())
                    suggestions.append(f"'{col}' → datetime 타입으로 변환 추천")
                except:
                    pass
            
            # 숫자로 변환 가능한지 체크
            try:
                pd.to_numeric(df[col], errors='raise')
                suggestions.append(f"'{col}' → 숫자 타입으로 변환 가능")
            except:
                pass
    
    if suggestions:
        for suggestion in suggestions:
            st.info(suggestion)
    else:
        st.success("현재 데이터 타입이 적절해 보입니다!")

else:
    st.info("📁 Google Drive 링크를 입력하거나 CSV 파일을 업로드하세요.")
