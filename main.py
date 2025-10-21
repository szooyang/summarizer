import textwrap
import streamlit as st
from openai import OpenAI

# ---------------------------
# 기본 설정
# ---------------------------
st.set_page_config(page_title="학생 프로젝트 보고서 요약기", page_icon="📝", layout="wide")
st.title("📝 학생 프로젝트 보고서 요약기")
st.caption("붙여넣은 1000자 보고서를 50/100/300/500자로 요약하고, 교사 질문 관점 요약도 생성합니다.")

# OpenAI 클라이언트 (Streamlit Cloud의 Secrets 사용)
# .streamlit/secrets.toml 에 openai_api_key="sk-..." 로 저장해 두세요.
client = OpenAI(api_key=st.secrets["openai_api_key"])

# ---------------------------
# 유틸 함수
# ---------------------------
def trim_to_chars(text: str, limit: int) -> str:
    """문장 자연스러움을 해치지 않도록 문자 수 제한 내로 자르기."""
    if len(text) <= limit:
        return text.strip()

    # 우선 잘라서 문장 끝(다./.!?) 기준으로 보정
    cut = text[:limit].rstrip()
    # 가장 마지막 문장종결부호 위치 찾기
    endings = ["다.", ".", "!", "?", "요.", "임."]
    last_end = -1
    for end in endings:
        pos = cut.rfind(end)
        if pos > last_end:
            last_end = pos + len(end)
    if last_end >= 10:  # 너무 초반에서 끊기면 어색하니 최소 길이 보장
        return cut[:last_end].strip()
    return cut.strip()

def summarize_with_limit(report: str, limit: int, teacher_hint: str | None = None) -> str:
    """OpenAI로 요약한 뒤, 최종적으로 문자 수 제한을 만족하도록 보정."""
    base_rules = (
        "규칙:\n"
        "1) 한국어로 한 단락으로만 작성\n"
        "2) 새로운 사실 추가 금지, 보고서 내용 기반 핵심만\n"
        "3) 활동 목적→주요 수행→성과/결과→배운 점(또는 다음 단계) 흐름을 가능하면 유지\n"
        "4) 숫자/지표/결과는 존재할 경우 명시적으로 포함\n"
        f"5) 공백 포함 {limit}자 이내 목표(자연스러운 문장 선호)\n"
    )

    perspective = ""
    if teacher_hint:
        perspective = (
            f"\n교사 질문 관점 요약 지시: '{teacher_hint}'의 관점에서 가장 관련 있는 내용만 선택해 요약.\n"
        )

    prompt = (
        "다음은 고등학생의 프로젝트 활동 보고서다. 지시를 따라 요약하라.\n\n"
        f"{base_rules}{perspective}\n"
        "[보고서 본문]\n"
        f"{report}\n\n"
        "출력은 불릿/번호 없이 한 단락으로만."
    )

    resp = client.responses.create(
        model="gpt-4o-mini",
        input=prompt,
        temperature=0.2,
    )
    text = resp.output_text
    return trim_to_chars(text, limit)

# ---------------------------
# 사이드바 옵션
# ---------------------------
with st.sidebar:
    st.header("⚙️ 옵션")
    temperature = st.slider("창의성(temperature)", 0.0, 1.0, 0.2, 0.05)
    st.caption("※ 기본은 정확한 요약을 위해 낮게 설정됩니다.")

# ---------------------------
# 본문 입력 영역
# ---------------------------
st.subheader("1) 1000자 보고서 붙여넣기")
report = st.text_area(
    "학생이 작성한 프로젝트 활동 보고서를 붙여넣어 주세요.",
    height=260,
    placeholder=(
        "예) 저희 팀은 지역 하천 수질을 주제로 프로젝트를 진행했습니다. 먼저 조사 설문을 만들고..."
    ),
)

colA, colB = st.columns([1, 1])

with colA:
    st.subheader("2) 자동 요약 (50/100/300/500자)")
    gen_default = st.button("요약 생성", use_container_width=True, type="primary")

with colB:
    st.subheader("3) 교사 질문 관점 요약")
    teacher_q = st.text_input(
        "교사 질문/관점 (예: '협업 과정에서의 역할 분담과 갈등 해결을 중심으로 요약')",
        placeholder="요약 관점을 입력한 뒤 '생성' 버튼을 누르세요.",
    )
    gen_q = st.button("질문 관점 요약 생성", use_container_width=True)

# temperature 반영을 위해 요약 함수 내부에서 사용할 수 있도록 재정의
def summarize_with_limit_temp(report: str, limit: int, teacher_hint: str | None = None) -> str:
    base_rules = (
        "규칙:\n"
        "1) 한국어로 한 단락으로만 작성\n"
        "2) 새로운 사실 추가 금지, 보고서 내용 기반 핵심만\n"
        "3) 활동 목적→주요 수행→성과/결과→배운 점(또는 다음 단계) 흐름을 가능하면 유지\n"
        "4) 숫자/지표/결과는 존재할 경우 명시적으로 포함\n"
        f"5) 공백 포함 {limit}자 이내 목표(자연스러운 문장 선호)\n"
    )
    perspective = ""
    if teacher_hint:
        perspective = (
            f"\n교사 질문 관점 요약 지시: '{teacher_hint}'의 관점에서 가장 관련 있는 내용만 선택해 요약.\n"
        )
    prompt = (
        "다음은 고등학생의 프로젝트 활동 보고서다. 지시를 따라 요약하라.\n\n"
        f"{base_rules}{perspective}\n"
        "[보고서 본문]\n"
        f"{report}\n\n"
        "출력은 불릿/번호 없이 한 단락으로만."
    )
    resp = client.responses.create(
        model="gpt-4o-mini",
        input=prompt,
        temperature=float(temperature),
    )
    return trim_to_chars(resp.output_text, limit)

# ---------------------------
# 결과 출력
# ---------------------------
if gen_default:
    if not report.strip():
        st.warning("보고서를 먼저 입력해 주세요.")
    else:
        tabs = st.tabs(["50자", "100자", "300자", "500자"])
        limits = [50, 100, 300, 500]
        for tab, limit in zip(tabs, limits):
            with tab:
                with st.spinner(f"{limit}자 요약 생성 중..."):
                    try:
                        summary = summarize_with_limit_temp(report, limit)
                        st.write(summary)
                        st.caption(f"문자 수: {len(summary)}")
                    except Exception as e:
                        st.error(f"요약 중 오류가 발생했습니다: {e}")

if gen_q:
    if not report.strip():
        st.warning("보고서를 먼저 입력해 주세요.")
    elif not teacher_q.strip():
        st.warning("교사 질문/관점을 입력해 주세요.")
    else:
        with st.spinner("질문 관점 요약 생성 중..."):
            try:
                # 관점 요약은 기본 300자/500자 두 가지로 제공
                q_limits = [300, 500]
                qt1, qt2 = st.tabs([f"관점 요약 {q_limits[0]}자", f"관점 요약 {q_limits[1]}자"])
                with qt1:
                    s1 = summarize_with_limit_temp(report, q_limits[0], teacher_hint=teacher_q)
                    st.write(s1)
                    st.caption(f"문자 수: {len(s1)}")
                with qt2:
                    s2 = summarize_with_limit_temp(report, q_limits[1], teacher_hint=teacher_q)
                    st.write(s2)
                    st.caption(f"문자 수: {len(s2)}")
            except Exception as e:
                st.error(f"요약 중 오류가 발생했습니다: {e}")

# 푸터
st.divider()
st.markdown(
    textwrap.dedent(
        """
        **사용 팁**
        - 보고서는 최대한 구체적으로 붙여넣을수록 요약 품질이 좋아집니다.
        - 질문 관점 예시: *"문제정의와 데이터 수집의 타당성에 초점"* / *"협업과 역할분담 중심"* / *"성과 지표와 한계, 다음 단계"*
        """
    )
)
