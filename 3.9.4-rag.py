from typing import Literal, Optional, List, Tuple
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    trim_messages,
)
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import os

Provider = Literal["openai", "ollama"]

# 세션 메모리 저장소
STORE = {}

TEMPERATURE = 0.3
TOP_P = 0.9
MAX_TOKENS = 8000

# RAG 관련 설정
PDF_PATH = "data/tax_faq.pdf"
VECTOR_STORE_PATH = "data/vector_store"
SIMILARITY_THRESHOLD = 0.6  # 유사도 임계값


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


def make_llm(provider: Provider = "openai", model: Optional[str] = None):
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


def make_message_trimmer(llm):
    return trim_messages(
        max_tokens=MAX_TOKENS,
        strategy="last",
        token_counter=llm,
        include_system=True,
        start_on="human",
    )


def load_pdf_and_create_vectorstore():
    """PDF를 로드하고 벡터 스토어를 생성합니다."""
    if os.path.exists(VECTOR_STORE_PATH):
        print("기존 벡터 스토어를 로드합니다...")
        return FAISS.load_local(
            VECTOR_STORE_PATH, OpenAIEmbeddings(), allow_dangerous_deserialization=True
        )

    print("PDF를 로드하고 벡터 스토어를 생성합니다...")
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    # 문서를 청크로 분할
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)

    # 벡터 스토어 생성
    vectorstore = FAISS.from_documents(splits, OpenAIEmbeddings())

    # 벡터 스토어 저장
    os.makedirs(os.path.dirname(VECTOR_STORE_PATH), exist_ok=True)
    vectorstore.save_local(VECTOR_STORE_PATH)

    return vectorstore


def analyze_intent(user_input: str, llm) -> bool:
    """사용자 입력이 연말정산 관련인지 분석합니다."""
    intent_prompt = f"""
    다음 질문이 연말정산, 세금, 소득세, 공제, 신고, 납부, 세무 관련 내용인지 분석해주세요.
    
    질문: {user_input}
    
    연말정산 관련 키워드: 연말정산, 소득세, 세금, 공제, 신고, 납부, 세무, 근로소득, 사업소득, 기타소득, 
    의료비공제, 교육비공제, 기부금공제, 연금보험료공제, 주택자금공제, 자녀세액공제, 
    국세청, 홈택스, 종합소득세, 부가가치세, 세법, 세율, 과세표준, 세액공제, 세액감면
    
    연말정산 관련이면 "YES", 그렇지 않으면 "NO"로만 답변해주세요.
    """

    response = llm.invoke([HumanMessage(content=intent_prompt)])
    return "YES" in response.content.upper()


def search_rag_documents(
    user_input: str, vectorstore, llm
) -> Tuple[List[Document], float]:
    """RAG를 사용하여 관련 문서를 검색합니다."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(user_input)

    # 유사도 점수 계산 - similarity_search_with_score 사용
    try:
        docs_with_scores = vectorstore.similarity_search_with_score(user_input, k=3)
        if docs_with_scores:
            similarity_score = max(0, 1 - docs_with_scores[0][1])
        else:
            similarity_score = 0.0
    except:
        similarity_score = 0.8 if docs else 0.0

    return docs, similarity_score


def generate_rag_response(user_input: str, docs: List[Document], llm) -> str:
    context = "\n\n".join([doc.page_content for doc in docs])

    rag_prompt = f"""
    다음 문서들을 참고하여 사용자의 질문에 답변해주세요.
    
    참고 문서:
    {context}
    
    사용자 질문: {user_input}
    
    답변 시 주의사항:
    - 문서에 있는 정보만을 바탕으로 답변하세요
    - 불확실한 내용은 추측하지 마세요
    - 한국어로 답변하세요
    - 구체적이고 정확한 정보를 제공하세요
    """

    response = llm.invoke([SystemMessage(content=rag_prompt)])
    return response.content


def make_chat_chain(llm):
    """
    trim_messages를 사용하여 메시지 히스토리를 제한하는 채팅 체인을 생성합니다.
    """
    trimmer = make_message_trimmer(llm)
    base = trimmer | llm

    return RunnableWithMessageHistory(base, get_session_history)


def process_user_query(user_input: str, llm, vectorstore, chat, cfg):
    """사용자 질문을 처리하는 메인 로직"""
    print("🔍 의도를 분석하는 중...")

    # 1. 의도 분석
    is_tax_related = analyze_intent(user_input, llm)

    if not is_tax_related:
        print("💬 일반 질문으로 처리합니다...")
        print("🤖 AI: ", end="", flush=True)
        response = chat.invoke([HumanMessage(content=user_input)], cfg)
        print(response.content)
        return

    print("📚 연말정산 관련 질문으로 인식하여 RAG 검색을 수행합니다...")

    # 2. RAG 검색
    docs, similarity_score = search_rag_documents(user_input, vectorstore, llm)
    print(f"📊 검색된 문서 수: {len(docs)}, 유사도 점수: {similarity_score:.2f}")

    # 3. 유사도 검증
    if similarity_score < SIMILARITY_THRESHOLD or not docs:
        print(
            "🤖 AI: 죄송합니다. 제공된 자료에서 해당 질문에 대한 충분한 정보를 찾을 수 없습니다. 더 구체적인 질문을 해주시거나 국세청 홈택스(https://hometax.go.kr)에서 확인해보시기 바랍니다."
        )
        return

    # 4. RAG 기반 답변 생성
    print("🤖 AI: ", end="", flush=True)
    rag_response = generate_rag_response(user_input, docs, llm)
    print(rag_response)


def main():
    provider = input("Provider 선택 (openai / ollama) [openai]: ").strip() or "openai"
    model = input("모델 이름 (비워두면 기본값): ").strip() or None

    llm = make_llm(provider=provider, model=model)
    chat = make_chat_chain(llm)
    cfg = {"configurable": {"session_id": "multi-chat-session"}}

    # RAG 시스템 초기화
    print("🚀 연말정산 RAG 시스템을 초기화합니다...")
    try:
        vectorstore = load_pdf_and_create_vectorstore()
        print("✅ RAG 시스템 초기화 완료!")
    except Exception as e:
        print(f"❌ RAG 시스템 초기화 실패: {e}")
        print("일반 채팅 모드로 진행합니다...")
        vectorstore = None

    print("\n\n💬 연말정산 간소화 질답 시스템에 오신 것을 환영합니다!")
    print("연말정산 관련 질문을 하시면 자료를 바탕으로 답변드립니다.")
    print("(종료하려면 'exit' 또는 'quit')")

    while True:
        user_input = input("\n👤 You: ")
        if user_input.lower() in {"exit", "quit"}:
            print("👋 종료합니다.")
            break

        if vectorstore:
            process_user_query(user_input, llm, vectorstore, chat, cfg)
        else:
            print("🤖 AI: ", end="", flush=True)
            response = chat.invoke([HumanMessage(content=user_input)], cfg)
            print(response.content)


if __name__ == "__main__":
    main()
