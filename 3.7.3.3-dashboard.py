import streamlit as st
import pandas as pd
from openai import OpenAI
import altair as alt
from collections import Counter

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

client = OpenAI()

st.set_page_config(page_title="리뷰 분석 대시보드", layout="wide")
st.title("🧠 리뷰 감정 분석 대시보드")

# 1. 데이터 로드
df = pd.read_csv("data/review/analyzed_reviews.csv")

# 2. 감정 분포
sentiment_count = df["sentiment"].value_counts().reset_index()
sentiment_count.columns = ["sentiment", "count"]
chart = (
    alt.Chart(sentiment_count)
    .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8)
    .encode(x="sentiment", y="count", color="sentiment")
)
st.subheader("📈 감정 분포")
st.altair_chart(chart, use_container_width=True)

# 3. 별점 평균 및 분포
st.subheader("⭐️ 평균 평점")
avg_rate = df["rate_review"].mean()
st.metric(label="평균 평점", value=f"{avg_rate:.2f} / 5")

rate_chart = alt.Chart(df).mark_bar().encode(x="rate_review:O", y="count()")
st.altair_chart(rate_chart, use_container_width=True)

# 4. 키워드 빈도
all_keywords = []
for k in df["keywords"].dropna():
    all_keywords.extend([x.strip() for x in k.split(",") if x.strip()])
keywords_df = pd.DataFrame(
    Counter(all_keywords).most_common(15), columns=["keyword", "count"]
)

st.subheader("🔑 상위 키워드 Top 15")
st.bar_chart(keywords_df.set_index("keyword"))

# 5. 요약 인사이트 (LLM)
llm = ChatOpenAI(model="gpt-4o-mini")
summary_prompt = f"""
아래 리뷰 데이터는 별점, 감정, 주요 키워드 정보를 담고 있습니다.
이 데이터를 기반으로 전반적인 소비자 반응과 개선점, 주요 특징을 5문장 이내로 요약해줘.
"""

prompt = ChatPromptTemplate.from_messages(
    [("system", summary_prompt), ("human", "{q}")]
)
chain = prompt | llm
summary = chain.invoke({"q": df.to_string(index=False)})

st.subheader("💡 AI 요약 인사이트")
st.write(summary.content)
