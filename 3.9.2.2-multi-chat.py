from typing import Literal, Optional
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

Provider = Literal["openai", "ollama"]

# 세션 메모리 저장소
STORE = {}

TEMPERATURE = 0.3
TOP_P = 0.9


def get_session_history(session_id: str):
    if session_id not in STORE:
        STORE[session_id] = ChatMessageHistory()
    return STORE[session_id]


def make_llm(provider: Provider = "ollama", model: Optional[str] = None):
    if provider == "openai":
        return ChatOpenAI(
            model=model or "gpt-4.1-mini",
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
    else:
        return ChatOllama(
            model=model or "llama3.2",
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )


def make_chat_chain(llm):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant. Answer in Korean by default."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )

    base = prompt | llm

    return RunnableWithMessageHistory(
        base,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )


def main():
    """
    메인 실행 함수: 사용자와의 대화형 채팅 인터페이스를 제공합니다.

    실행 과정:
    1. 사용자로부터 AI 제공업체와 모델 선택
    2. 선택된 설정으로 LLM과 채팅 체인 초기화
    3. 대화 루프 실행 (사용자가 종료할 때까지)
    4. 각 대화는 세션 기록에 저장되어 맥락 유지
    """
    provider = input("Provider 선택 (openai / ollama) [ollama]: ").strip() or "ollama"
    model = input("모델 이름 (비워두면 기본값): ").strip() or None

    llm = make_llm(provider=provider, model=model)
    chat = make_chat_chain(llm)
    cfg = {"configurable": {"session_id": "multi-chat-session"}}

    print("\n대화를 시작하세요! (종료하려면 'exit' 또는 'quit')")

    while True:
        user_input = input("👤 You: ")

        if user_input.lower() in {"exit", "quit"}:
            print("👋 종료합니다.")
            break

        res = chat.invoke({"input": user_input}, cfg)
        print(f"🤖 AI: {res.content}\n")


if __name__ == "__main__":
    main()
