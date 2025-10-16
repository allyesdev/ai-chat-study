"""
🤖 Agent 스트리밍 버전 (04-agent-stream.py)

📋 Agent에서 스트리밍을 활용하는 방법들:

1. 🔍 Tool 실행 과정 실시간 표시
2. 🧠 Agent의 사고 과정 단계별 출력
3. 📊 중간 결과물들 순차적 표시
4. ⚡ 사용자에게 진행 상황 피드백

Agent 스트리밍의 장점:
- 각 도구 실행 과정을 실시간으로 확인
- Agent의 의사결정 과정 투명화
- 긴 작업의 진행 상황 모니터링
- 오류 발생 지점 빠른 파악
"""

from typing import Literal, Optional
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
import time

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
            streaming=True,  # 🔥 Agent에서도 스트리밍 활성화
        )
    else:
        return ChatOllama(
            model=model or "llama3.2",
            temperature=TEMPERATURE,
            top_p=TOP_P,
            streaming=True,  # 🔥 Agent에서도 스트리밍 활성화
        )


# 🔧 예시 도구들 정의
@tool
def calculate(expression: str) -> str:
    """수학 계산을 수행합니다. 예: calculate('2 + 2')"""
    try:
        result = eval(expression)
        return f"계산 결과: {result}"
    except Exception as e:
        return f"계산 오류: {e}"


@tool
def get_weather(city: str) -> str:
    """도시의 날씨 정보를 가져옵니다."""
    # 실제로는 API 호출하지만 여기서는 시뮬레이션
    time.sleep(1)  # 네트워크 지연 시뮬레이션
    return f"{city}의 날씨: 맑음, 22°C"


@tool
def search_web(query: str) -> str:
    """웹에서 정보를 검색합니다."""
    time.sleep(2)  # 검색 지연 시뮬레이션
    return f"'{query}'에 대한 검색 결과: 관련 정보를 찾았습니다."


def stream_agent_execution(agent_executor, user_input: str):
    """
    🔥 Agent 실행 과정을 스트리밍으로 표시

    Agent의 각 단계별 실행 과정을 실시간으로 보여줍니다:
    1. Agent의 사고 과정 (Reasoning)
    2. 도구 선택 및 실행
    3. 중간 결과들
    4. 최종 응답
    """
    print("🤖 Agent가 작업을 시작합니다...\n")

    # 🔥 Agent 실행 과정을 스트리밍으로 처리
    for chunk in agent_executor.stream({"input": user_input}):
        # 🔍 각 청크의 타입과 내용 분석
        if "agent" in chunk:
            # 🧠 Agent의 사고 과정
            agent_output = chunk["agent"]
            if hasattr(agent_output, "messages") and agent_output.messages:
                for message in agent_output.messages:
                    if hasattr(message, "content"):
                        print(f"💭 Agent 사고: {message.content}")
                        print("-" * 50)

        elif "tools" in chunk:
            # 🔧 도구 실행 과정
            tool_output = chunk["tools"]
            if hasattr(tool_output, "messages") and tool_output.messages:
                for message in tool_output.messages:
                    if hasattr(message, "content"):
                        print(f"🔧 도구 실행: {message.content}")
                        print("-" * 50)

        elif "output" in chunk:
            # 📤 최종 출력
            print(f"✅ 최종 결과: {chunk['output']}")
            print("=" * 50)


def create_agent_with_tools(llm):
    """도구가 포함된 Agent 생성"""
    tools = [calculate, get_weather, search_web]

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """당신은 도구를 사용할 수 있는 도움이 되는 AI 어시스턴트입니다.
        
사용 가능한 도구들:
- calculate: 수학 계산
- get_weather: 날씨 정보 조회
- search_web: 웹 검색

각 단계별로 무엇을 하고 있는지 명확히 설명하고, 
도구를 사용할 때는 왜 그 도구를 선택했는지 이유를 설명하세요.
한국어로 응답하세요.""",
            ),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agent = create_openai_tools_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)


def main():
    """Agent 스트리밍 데모"""
    print("🤖 Agent 스트리밍 데모")
    print("=" * 50)

    provider = input("Provider 선택 (openai / ollama) [ollama]: ").strip() or "ollama"
    model = input("모델 이름 (비워두면 기본값): ").strip() or None

    llm = make_llm(provider=provider, model=model)
    agent_executor = create_agent_with_tools(llm)

    print("\n사용 가능한 도구들:")
    print("- calculate: 수학 계산")
    print("- get_weather: 날씨 정보")
    print("- search_web: 웹 검색")
    print("\n대화를 시작하세요! (종료하려면 'exit' 또는 'quit')")

    while True:
        user_input = input("\n👤 You: ")

        if user_input.lower() in {"exit", "quit"}:
            print("👋 종료합니다.")
            break

        # 🔥 Agent 실행 과정을 스트리밍으로 표시
        stream_agent_execution(agent_executor, user_input)


if __name__ == "__main__":
    main()
