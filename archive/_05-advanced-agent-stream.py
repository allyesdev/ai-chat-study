"""
🚀 고급 Agent 스트리밍 (05-advanced-agent-stream.py)

📋 고급 Agent 스트리밍 기능들:

1. 🔄 단계별 진행 상황 표시
2. ⏱️ 각 단계별 소요 시간 측정
3. 🎯 도구 실행 결과 실시간 표시
4. 📊 전체 작업 진행률 표시
5. 🚨 오류 발생 시 즉시 알림
"""

from typing import Literal, Optional, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
import time
import json
from datetime import datetime

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
            streaming=True,
        )
    else:
        return ChatOllama(
            model=model or "llama3.2",
            temperature=TEMPERATURE,
            top_p=TOP_P,
            streaming=True,
        )


# 🔧 고급 도구들 정의
@tool
def analyze_data(data: str) -> str:
    """데이터를 분석하고 통계를 제공합니다."""
    time.sleep(1.5)  # 분석 시간 시뮬레이션
    return f"데이터 분석 완료: {len(data)}개 항목, 평균 길이: {len(data)/10:.1f}"


@tool
def generate_report(topic: str, data: str) -> str:
    """주제와 데이터를 바탕으로 보고서를 생성합니다."""
    time.sleep(2)  # 보고서 생성 시간 시뮬레이션
    return f"'{topic}' 주제의 보고서가 생성되었습니다. (데이터: {data[:50]}...)"


@tool
def send_notification(message: str) -> str:
    """알림을 전송합니다."""
    time.sleep(0.5)
    return f"알림 전송 완료: {message}"


@tool
def complex_calculation(operation: str, numbers: str) -> str:
    """복잡한 수학 연산을 수행합니다."""
    time.sleep(1)
    try:
        # 간단한 연산 시뮬레이션
        num_list = [float(x.strip()) for x in numbers.split(",")]
        if operation == "sum":
            result = sum(num_list)
        elif operation == "average":
            result = sum(num_list) / len(num_list)
        elif operation == "max":
            result = max(num_list)
        elif operation == "min":
            result = min(num_list)
        else:
            result = "지원하지 않는 연산"

        return f"{operation} 연산 결과: {result}"
    except Exception as e:
        return f"계산 오류: {e}"


class StreamingAgentMonitor:
    """🔥 Agent 실행 과정을 모니터링하는 클래스"""

    def __init__(self):
        self.start_time = None
        self.step_count = 0
        self.tool_calls = []
        self.errors = []

    def start_monitoring(self):
        """모니터링 시작"""
        self.start_time = time.time()
        self.step_count = 0
        self.tool_calls = []
        self.errors = []
        print("🚀 Agent 모니터링 시작")
        print("=" * 60)

    def log_step(self, step_name: str, details: str = ""):
        """단계별 로깅"""
        self.step_count += 1
        elapsed = time.time() - self.start_time if self.start_time else 0
        print(f"📋 단계 {self.step_count}: {step_name}")
        if details:
            print(f"   💡 {details}")
        print(f"   ⏱️  경과 시간: {elapsed:.2f}초")
        print("-" * 40)

    def log_tool_call(self, tool_name: str, input_data: str, result: str):
        """도구 호출 로깅"""
        self.tool_calls.append(
            {
                "tool": tool_name,
                "input": input_data,
                "result": result,
                "timestamp": datetime.now().strftime("%H:%M:%S"),
            }
        )
        print(f"🔧 도구 실행: {tool_name}")
        print(f"   📥 입력: {input_data[:50]}...")
        print(f"   📤 결과: {result[:100]}...")
        print("-" * 40)

    def log_error(self, error: str):
        """오류 로깅"""
        self.errors.append(error)
        print(f"🚨 오류 발생: {error}")
        print("-" * 40)

    def finish_monitoring(self):
        """모니터링 종료 및 요약"""
        total_time = time.time() - self.start_time if self.start_time else 0
        print("=" * 60)
        print("📊 Agent 실행 요약")
        print(f"   ⏱️  총 소요 시간: {total_time:.2f}초")
        print(f"   📋 총 단계 수: {self.step_count}")
        print(f"   🔧 도구 호출 수: {len(self.tool_calls)}")
        print(f"   🚨 오류 수: {len(self.errors)}")

        if self.tool_calls:
            print("\n🔧 사용된 도구들:")
            for i, call in enumerate(self.tool_calls, 1):
                print(f"   {i}. {call['tool']} ({call['timestamp']})")

        if self.errors:
            print("\n🚨 발생한 오류들:")
            for i, error in enumerate(self.errors, 1):
                print(f"   {i}. {error}")


def stream_advanced_agent_execution(agent_executor, user_input: str):
    """🔥 고급 Agent 스트리밍 실행"""
    monitor = StreamingAgentMonitor()
    monitor.start_monitoring()

    try:
        # 🔥 Agent 실행 과정을 스트리밍으로 처리
        for chunk in agent_executor.stream({"input": user_input}):
            if "agent" in chunk:
                # 🧠 Agent의 사고 과정
                agent_output = chunk["agent"]
                if hasattr(agent_output, "messages") and agent_output.messages:
                    for message in agent_output.messages:
                        if hasattr(message, "content"):
                            monitor.log_step("Agent 사고 과정", message.content)

            elif "tools" in chunk:
                # 🔧 도구 실행 과정
                tool_output = chunk["tools"]
                if hasattr(tool_output, "messages") and tool_output.messages:
                    for message in tool_output.messages:
                        if hasattr(message, "content"):
                            # 도구 이름과 입력 추출 (실제로는 더 정교한 파싱 필요)
                            monitor.log_tool_call(
                                "도구", "입력 데이터", message.content
                            )

            elif "output" in chunk:
                # 📤 최종 출력
                monitor.log_step("최종 결과 생성", "Agent가 최종 답변을 생성했습니다.")
                print(f"✅ 최종 결과: {chunk['output']}")

    except Exception as e:
        monitor.log_error(str(e))

    finally:
        monitor.finish_monitoring()


def create_advanced_agent_with_tools(llm):
    """고급 도구가 포함된 Agent 생성"""
    tools = [analyze_data, generate_report, send_notification, complex_calculation]

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """당신은 고급 도구를 사용할 수 있는 AI 어시스턴트입니다.

사용 가능한 도구들:
- analyze_data: 데이터 분석 및 통계 제공
- generate_report: 보고서 생성
- send_notification: 알림 전송
- complex_calculation: 복잡한 수학 연산

각 단계별로 진행 상황을 명확히 설명하고,
도구 사용 시에는 입력과 결과를 자세히 설명하세요.
한국어로 응답하세요.""",
            ),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agent = create_openai_tools_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)


def main():
    """고급 Agent 스트리밍 데모"""
    print("🚀 고급 Agent 스트리밍 데모")
    print("=" * 60)

    provider = input("Provider 선택 (openai / ollama) [ollama]: ").strip() or "ollama"
    model = input("모델 이름 (비워두면 기본값): ").strip() or None

    llm = make_llm(provider=provider, model=model)
    agent_executor = create_advanced_agent_with_tools(llm)

    print("\n사용 가능한 고급 도구들:")
    print("- analyze_data: 데이터 분석")
    print("- generate_report: 보고서 생성")
    print("- send_notification: 알림 전송")
    print("- complex_calculation: 복잡한 수학 연산")
    print("\n대화를 시작하세요! (종료하려면 'exit' 또는 'quit')")

    while True:
        user_input = input("\n👤 You: ")

        if user_input.lower() in {"exit", "quit"}:
            print("👋 종료합니다.")
            break

        # 🔥 고급 Agent 실행 과정을 스트리밍으로 표시
        stream_advanced_agent_execution(agent_executor, user_input)


if __name__ == "__main__":
    main()
