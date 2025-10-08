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
            streaming=True,  # 🔥 스트리밍 모드 활성화 - 응답을 실시간으로 받아옴
        )
    else:
        return ChatOllama(
            model=model or "llama3.2",
            temperature=TEMPERATURE,
            top_p=TOP_P,
            streaming=True,  # 🔥 스트리밍 모드 활성화 - 응답을 실시간으로 받아옴
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


def stream_reply(*, chat, user_input, cfg):
    """
    사용자 입력에 대한 스트리밍 응답을 처리합니다.

    Args:
        chat: 채팅 체인
        user_input: 사용자 입력
        cfg: 설정

    Note:
        - 스트리밍 응답을 처리하고 출력합니다.
    """
    for chunk in chat.stream({"input": user_input}, cfg):
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
