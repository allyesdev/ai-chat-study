# AI Chat 프로젝트

LangChain을 활용한 다양한 AI 채팅 시스템 구현 프로젝트입니다. OpenAI와 Ollama 모델을 지원하며, 기본 채팅부터 RAG(Retrieval-Augmented Generation) 시스템, Agent 기반 대화까지 다양한 기능을 제공합니다.

## 🚀 주요 기능

### 1. 기본 채팅 시스템

- **단일 질문-답변** (`3.9.2.1-simple-chat.py`)
- **다중 대화** (`3.9.2.2-multi-chat.py`) - 세션 히스토리 유지
- **스트리밍 채팅** (`3.9.2.3-multi-chat-stream.py`) - 실시간 응답 스트리밍

### 2. 메시지 관리

- **메시지 트리밍** (`3.9.2.4-trim-messages.py`) - 토큰 제한 내에서 대화 유지

### 3. RAG (Retrieval-Augmented Generation) 시스템

- **연말정산 질답 시스템** (`3.9.4-rag.py`)
- PDF 문서 기반 벡터 검색
- 의도 분석을 통한 스마트 답변

## 📋 요구사항

### 환경 설정

1. `.env` 파일 생성
2. OpenAI API 키 설정 (선택사항)
3. Ollama 설치 및 모델 다운로드 (로컬 실행용)

```bash
# .env 파일 예시
OPENAI_API_KEY=your_openai_api_key_here
```

## 🛠️ 설치 및 실행

### 1. 가상환경 설정

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 채팅 실행 시

VS code 실행 메뉴에서 클릭하여 실행 가능

## 📁 프로젝트 구조

```
ai-chat/
├── 3.9.2.1-simple-chat.py          # 단일 질문-답변
├── 3.9.2.2-multi-chat.py           # 다중 대화
├── 3.9.2.3-multi-chat-stream.py    # 스트리밍 채팅
├── 3.9.2.4-trim-messages.py        # 메시지 트리밍
├── 3.9.4-rag.py                    # RAG 시스템
├── data/
│   └── tax_faq.pdf                 # 연말정산 FAQ PDF
└── requirements.txt                # Python 패키지 목록
```

---

**LangChain**을 활용한 AI 채팅 시스템의 다양한 구현 방법을 학습하고 실험할 수 있는 프로젝트입니다. 🚀
