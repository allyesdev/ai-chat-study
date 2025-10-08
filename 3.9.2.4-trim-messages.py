from typing import Literal, Optional
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    trim_messages,
)

Provider = Literal["openai", "ollama"]

# 세션 메모리 저장소
STORE = {}

TEMPERATURE = 0.3
TOP_P = 0.9
MAX_TOKENS = 45


def get_session_history(session_id: str):
    if session_id not in STORE:
        message_history = ChatMessageHistory()
        message_history.add_message(
            SystemMessage(
                content="You are a helpful assistant. Answer in Korean by default."
            )
        )
        STORE[session_id] = message_history
    return STORE[session_id]


def make_llm(provider: Provider = "ollama", model: Optional[str] = None):
    if provider == "openai":
        return ChatOpenAI(
            model=model or "gpt-4.1-mini",
            temperature=TEMPERATURE,
            top_p=TOP_P,
            streaming=True,  # 🔥 스트리밍 모드 활성화 - 응답을 실시간으로 받아옴
        )
    else:
        return ChatOllama(
            model=model or "llama3.2",
            temperature=TEMPERATURE,
            top_p=TOP_P,
            streaming=True,  # 🔥 스트리밍 모드 활성화 - 응답을 실시간으로 받아옴
        )


def make_message_trimmer(llm):
    return trim_messages(
        max_tokens=MAX_TOKENS,
        strategy="last",
        token_counter=llm,
        include_system=True,
        start_on="human",
    )


def make_chat_chain(llm):
    """
    trim_messages를 사용하여 메시지 히스토리를 제한하는 채팅 체인을 생성합니다.
    """
    trimmer = make_message_trimmer(llm)
    base = trimmer | llm

    return RunnableWithMessageHistory(base, get_session_history)


def stream_reply(*, chat, user_input, cfg):
    for chunk in chat.stream([HumanMessage(content=user_input)], cfg):
        text = getattr(chunk, "content", None)
        if text is None and isinstance(chunk, dict):
            text = chunk.get("output_text") or chunk.get("content") or ""
        if text is None:
            text = str(chunk)
        print(text, end="", flush=True)
    print("\n")


def main():
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

        print("🤖 AI: ", end="", flush=True)
        stream_reply(chat=chat, user_input=user_input, cfg=cfg)


if __name__ == "__main__":
    main()
