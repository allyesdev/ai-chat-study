from typing import Literal, Optional, List
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

Provider = Literal["openai", "ollama"]

# 세션 메모리 저장소
STORE = {}

TEMPERATURE = 0.3
TOP_P = 0.9
MAX_MESSAGES_TO_KEEP = 2  # 최근 2개 메시지만 유지


def get_session_history(session_id: str):
    if session_id not in STORE:
        STORE[session_id] = ChatMessageHistory()
    return STORE[session_id]


def make_llm(provider: Provider = "ollama", model: Optional[str] = None):
    if provider == "openai":
        return ChatOpenAI(
            model=model or "gpt-4o-mini",
            temperature=TEMPERATURE,
            top_p=TOP_P,
            streaming=True,
        )
    else:
        return ChatOllama(
            model=model or "llama3.2",
            temperature=TEMPERATURE,
            top_p=TOP_P,
            streaming=True,
        )


def summarize_messages(messages: List, llm) -> str:
    """
    메시지들을 요약합니다.
    """
    if not messages:
        return ""

    # 메시지들을 텍스트로 변환
    conversation_text = ""
    for msg in messages:
        if isinstance(msg, HumanMessage):
            conversation_text += f"사용자: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            conversation_text += f"AI: {msg.content}\n"

    # 요약 프롬프트
    summarize_prompt = f"""
다음 대화를 간단히 요약해주세요. 핵심 내용만 2-3문장으로 정리해주세요.

대화 내용:
{conversation_text}

요약:
"""

    try:
        response = llm.invoke(summarize_prompt)
        return response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        print(f"요약 중 오류 발생: {e}")
        return "이전 대화 내용을 요약할 수 없습니다."


def summarize_messages(messages, llm):
    """
    메시지 히스토리를 정리하고 오래된 메시지들을 요약합니다.
    """

    if len(messages) <= MAX_MESSAGES_TO_KEEP + 1:
        return

    # 시스템 메시지와 최근 2개 메시지 분리
    system_messages = [msg for msg in messages if isinstance(msg, SystemMessage)]
    recent_messages = messages[-MAX_MESSAGES_TO_KEEP:]
    old_messages = messages[:-MAX_MESSAGES_TO_KEEP]  # 시스템 메시지 제외

    # 오래된 메시지가 있으면 요약
    if old_messages:
        summary = summarize_messages(old_messages, llm)
        summary_message = SystemMessage(content=f"[이전 대화 요약] {summary}")

        # 히스토리 재구성
        new_messages = system_messages + [summary_message] + recent_messages
        messages.clear()
        for msg in new_messages:
            messages.append(msg)


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


def stream_reply(*, chat, user_input, cfg, llm):
    """
    사용자 입력에 대한 스트리밍 응답을 처리하고 메시지 히스토리를 관리합니다.
    """
    # 현재 세션의 히스토리 가져오기
    session_id = cfg["configurable"]["session_id"]
    history = get_session_history(session_id)

    # 메시지 히스토리 정리 및 요약
    summarize_messages(history.messages, llm)

    # 스트리밍 응답 처리
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
    cfg = {"configurable": {"session_id": "summarize-chat-session"}}

    print("\n대화를 시작하세요! (종료하려면 'exit' 또는 'quit')")
    print("💡 최근 2개 메시지만 유지하고 이전 메시지들은 자동으로 요약됩니다.")

    while True:
        user_input = input("👤 You: ")
        if user_input.lower() in {"exit", "quit"}:
            print("👋 종료합니다.")
            break

        print("🤖 AI: ", end="", flush=True)
        stream_reply(chat=chat, user_input=user_input, cfg=cfg, llm=llm)


if __name__ == "__main__":
    main()
