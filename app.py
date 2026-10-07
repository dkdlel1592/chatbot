"""DATA 폴더의 PDF를 기반으로 답변하는 Streamlit RAG 챗봇입니다."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

# .env 파일의 OPENAI_API_KEY를 환경 변수로 불러옵니다.
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "DATA"


def get_openai_api_key() -> str | None:
    """Cloud Secrets 또는 로컬 .env에서 OpenAI API 키를 가져옵니다."""
    try:
        # Streamlit Cloud의 Secrets에 저장한 값을 가장 먼저 사용합니다.
        return st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    except FileNotFoundError:
        # 로컬에서 secrets.toml이 없을 때는 .env만 확인합니다.
        return os.getenv("OPENAI_API_KEY")


@st.cache_resource(show_spinner="PDF 문서를 읽고 검색 색인을 만드는 중입니다...")
def build_retriever():
    """DATA의 모든 PDF를 읽어 메모리 벡터 검색기로 만듭니다."""
    documents: list[Document] = []

    # 하위 폴더까지 포함해 모든 PDF 파일을 찾습니다.
    for pdf_path in sorted(DATA_DIR.rglob("*.pdf")):
        reader = PdfReader(pdf_path)
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"source": pdf_path.name, "page": page_number},
                    )
                )

    if not documents:
        raise ValueError("DATA 폴더에서 텍스트를 추출할 수 있는 PDF를 찾지 못했습니다.")

    # 긴 페이지를 작은 단위로 나누어 질문과 관련된 부분을 더 정확히 찾습니다.
    splitter = RecursiveCharacterTextSplitter(chunk_size=1_000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    # 요청한 임베딩 모델과 메모리 기반 벡터 DB를 사용합니다.
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = InMemoryVectorStore(embedding=embeddings)
    vector_store.add_documents(chunks)
    return vector_store.as_retriever(search_kwargs={"k": 4})


def format_context(documents: list[Document]) -> str:
    """검색된 문서를 LLM에 전달할 수 있는 문맥 문자열로 변환합니다."""
    return "\n\n".join(
        f"[파일: {doc.metadata['source']}, 페이지: {doc.metadata['page']}]\n{doc.page_content}"
        for doc in documents
    )


def answer_question(question: str, retriever) -> tuple[str, list[Document]]:
    """검색 결과만 근거로 답변을 생성하고, 함께 보여줄 출처를 반환합니다."""
    source_documents = retriever.invoke(question)
    context = format_context(source_documents)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """당신은 공무원 여비 관련 문서를 안내하는 도우미입니다.
반드시 제공된 문서 내용만 근거로 한국어로 답변하세요.
문서에 없는 정보이거나 근거가 부족하면 반드시 '제공된 문서에서 확인할 수 없습니다.'라고 답하세요.
추측, 일반 상식, 외부 지식을 덧붙이지 마세요.""",
            ),
            ("human", "문서 내용:\n{context}\n\n질문: {question}"),
        ]
    )

    # 최신 LangChain Runnable 방식으로 프롬프트, 모델, 출력 파서를 연결합니다.
    chain = prompt | ChatOpenAI(model="gpt-4o-mini", temperature=0) | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    return answer, source_documents


def verify_answer(answer: str, documents: list[Document]) -> bool:
    """답변의 모든 사실 주장이 검색된 문서에 근거하는지 확인합니다."""
    context = format_context(documents)
    verification_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """당신은 답변 검증자입니다.
제공된 문서만 근거로 답변을 검토하세요.
답변의 모든 사실 주장이 문서에 직접 뒷받침되고, 문서에 없는 추측이나 일반 지식이 없으면 VERIFIED만 출력하세요.
조금이라도 근거가 부족하거나 문서와 다른 내용이 있으면 NOT_VERIFIED만 출력하세요.
다른 설명은 절대 덧붙이지 마세요.""",
            ),
            ("human", "문서 내용:\n{context}\n\n검증할 답변:\n{answer}"),
        ]
    )
    verification_chain = (
        verification_prompt
        | ChatOpenAI(model="gpt-4o-mini", temperature=0)
        | StrOutputParser()
    )
    result = verification_chain.invoke({"context": context, "answer": answer})
    return result.strip() == "VERIFIED"


def show_sources(documents: list[Document]) -> None:
    """검색에 사용한 원문 발췌를 파일명과 페이지 번호로 표시합니다."""
    st.markdown("#### 출처와 근거 문장")
    for index, document in enumerate(documents, start=1):
        source = document.metadata["source"]
        page = document.metadata["page"]
        # 원문 그대로 보여 주되, 화면이 너무 길어지지 않도록 길이를 제한합니다.
        evidence = " ".join(document.page_content.split())
        if len(evidence) > 500:
            evidence = f"{evidence[:500]}..."
        st.markdown(f"**{index}. {source} (p. {page})**")
        st.caption(f"근거 문장(발췌): {evidence}")


def main() -> None:
    st.set_page_config(page_title="공무원 여비 RAG 챗봇", page_icon="📚")
    st.title("📚 공무원 여비 RAG 챗봇")
    st.write("DATA 폴더의 문서만 검색해 답변합니다.")

    if not get_openai_api_key():
        st.error(
            "로컬 .env 또는 Streamlit Cloud Secrets에 OPENAI_API_KEY를 설정해 주세요."
        )
        st.stop()

    try:
        retriever = build_retriever()
    except Exception as error:
        st.error(f"문서를 준비하는 중 오류가 발생했습니다: {error}")
        st.stop()

    question = st.chat_input("공무원 여비에 관해 질문해 보세요")
    if not question:
        return

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("문서를 검색하고 답변을 검증하고 있습니다..."):
            try:
                answer, source_documents = answer_question(question, retriever)
                is_verified = verify_answer(answer, source_documents)
            except Exception as error:
                st.error(f"답변을 생성하는 중 오류가 발생했습니다: {error}")
                return
        if is_verified:
            st.write(answer)
        else:
            st.warning(
                "제공된 문서에서 답변을 뒷받침할 충분한 근거를 확인하지 못했습니다."
            )
        show_sources(source_documents)


if __name__ == "__main__":
    main()
