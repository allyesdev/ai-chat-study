import pandas as pd
from dotenv import load_dotenv
from typing import Literal, Optional
from typing import List
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama


llm = ChatOpenAI(model="gpt-4o-mini")
# llm = ChatOllama(model="llama3.2")


class AnalyzeResult(BaseModel):
    sentiment: str = Field(description="감정(긍정/부정/중립)")
    keywords: List[str] = Field(description="주요 키워드")


# 1. 리뷰 데이터 로드
df = pd.read_csv("data/review/reviews.csv")


# 2. LLM 요청 함수
def analyze_review(review, rate_review):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
    아래 리뷰의 감정(긍정/부정/중립)을 분류하고, 주요 키워드 3개를 추출해줘. 주요 키워드는 리뷰를 쓴 사람이 느낀 점을 중심으로 추출해줘.
    결과는 JSON 형식으로 반환해.

    출력 예시: {{"sentiment": "긍정", "keywords": ["재미", "연출", "연기", "감동"]}}
    """,
            ),
            ("human", "{q}"),
        ]
    )

    chain = prompt | llm.with_structured_output(AnalyzeResult)

    res = chain.invoke({"q": f"리뷰: {review}\n별점: {rate_review}"})
    return res


# 3. 전체 리뷰 분석
results = []
max_reviews = 10
for i, row in df.iterrows():
    if i >= max_reviews:
        break
    text_review = row["text_review"]
    rate_review = row["rate_review"]
    r = analyze_review(text_review, rate_review)
    results.append(
        {
            "text_review": text_review,
            "rate_review": rate_review,
            "sentiment": r.sentiment,
            "keywords": ",".join(r.keywords),
        }
    )

analyzed_df = pd.DataFrame(results)
analyzed_df.to_csv("data/review/analyzed_reviews.csv", index=False)

print("✅ 감정 및 키워드 분석 완료 → data/review/analyzed_reviews.csv 저장됨")
