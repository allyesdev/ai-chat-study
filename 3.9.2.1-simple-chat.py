from dotenv import load_dotenv
from typing import Literal, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.language_models.chat_models import BaseChatModel

load_dotenv(dotenv_path=".env")

Provider = Literal["openai", "ollama"]

TEMPERATURE = 0.3
TOP_P = 0.9


def make_llm(provider: Provider = "ollama", model: Optional[str] = None):
    if provider == "openai":
        return ChatOpenAI(
            model=model or "gpt-5-mini",  # 원하는 OpenAI 모델명
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
    else:
        return ChatOllama(
            model=model or "llama3.2",  # 로컬에 pull한 모델명
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )


def ask_once(question: str, llm: BaseChatModel):
    prompt = ChatPromptTemplate.from_messages(
        [("system", "You are a concise assistant."), ("human", "{q}")]
    )
    chain = prompt | llm
    response = chain.invoke({"q": question})
    print(response)
    return response.content


if __name__ == "__main__":
    llm = make_llm(provider="ollama", model="llama3.2")
    # llm = make_llm(provider="openai", model="gpt-5")
    print(ask_once("가장 많이 사용되는 프로그래밍 언어는 뭐야?", llm))
