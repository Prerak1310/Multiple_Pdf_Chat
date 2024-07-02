import os
from dotenv import load_dotenv

load_dotenv()
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


def get_pdf_text(pdf_docs):  # function to get raw text from pages in multiple PDFs
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text = text + page.extract_text()
    return text


def get_text_chunks(raw_text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=6000,
        chunk_overlap=1000,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks


def get_vectorstore(text_chunks):
    vectorstore = FAISS.from_texts(text_chunks, HuggingFaceEmbeddings())
    return vectorstore


groq_api_key = os.getenv("GROQ_API_KEY_GIT")
llm = ChatGroq(
    model="llama3-8b-8192",
    api_key=groq_api_key,
)


def get_conversation_chain(vectorstore):

    # google_api_key = os.getenv("GOOGLE_API_KEY")
    # llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    # genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

    retriever = vectorstore.as_retriever()

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(
        llm,
        qa_prompt,
    )

    rag_chain = create_retrieval_chain(
        history_aware_retriever,
        question_answer_chain,
    )
    return rag_chain


def main():

    st.set_page_config(
        page_title="Chat with multiple PDFs",
        page_icon=":books:",
    )
    st.title("Chat with multiple PDFs :books:")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "raw_text" not in st.session_state:
        st.session_state["raw_text"] = ""

    pdf_docs = st.file_uploader(  # pdf_docs will be of type "list"
        "Upload your PDFs here and click on 'process'",
        accept_multiple_files=True,
        disabled=False,
    )
    if not pdf_docs:
        st.info("Kindly upload a pdf to chat :)")
    if st.button("Process"):
        if pdf_docs:

            with st.spinner(
                "Kindly wait while we process your documents"
            ):  # all processes within spinner will run while the user sees the processing animation

                # get pdf text
                st.session_state.raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(st.session_state.raw_text)
                # st.write(text_chunks)

                # create vector store with embeddings
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.rag_chain = get_conversation_chain(vectorstore)
            st.toast("You are ready to chat", icon="🎉")
        else:
            print(st.session_state.raw_text)
            st.toast("Kindly enter a pdf", icon="⚠️")
    for i in range(0, len(st.session_state.chat_history)):
        if i & 1:
            with st.chat_message("user"):
                st.markdown(st.session_state.chat_history[i].content)
        else:
            with st.chat_message("assistant"):
                st.markdown(st.session_state.chat_history[i].content)

    if st.session_state.raw_text != "" and pdf_docs:

        user_input = st.chat_input("Ask a question about your documents:")
        if user_input:
            with st.chat_message("user"):
                st.markdown(user_input)

            try:
                response = st.session_state.rag_chain.invoke(
                    {"input": user_input, "chat_history": st.session_state.chat_history}
                )
                with st.chat_message("assistant"):
                    st.markdown(response["answer"])
                st.session_state.chat_history.extend(
                    [
                        HumanMessage(content=user_input),
                        AIMessage(content=response["answer"]),
                    ]
                )
            except:
                st.error("An error occurred")


if __name__ == "__main__":
    main()
