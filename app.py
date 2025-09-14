import os

from dotenv import load_dotenv

load_dotenv()
import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from PyPDF2 import PdfReader


# function to get raw text from pages in multiple PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text = text + page.extract_text()
    return text


# function to break the raw text into chunks
def get_text_chunks(raw_text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=6000,
        chunk_overlap=1000,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks


# function to return a vectorstore with embeddings of the the chunks
def get_vectorstore(text_chunks):
    if text_chunks == []:
        st.error("The above file couln't be processed", icon="⚠️")
        return
    vectorstore = FAISS.from_texts(text_chunks, HuggingFaceEmbeddings())
    return vectorstore


# setting up the chatbot
groq_api_key = os.getenv("GROQ_API_KEY_GIT")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=groq_api_key)


# setting up the RAG chain
def get_conversation_chain(vectorstore):
    if vectorstore:
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
            "the question. If the question is not related to the context provided,say that you "
            "don't know the answer. Use three sentences maximum and keep the "
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
        st.toast("You are ready to chat", icon="🎉")
        return rag_chain
    else:
        return


def main():
    # setting up page title
    st.set_page_config(
        page_title="Chat with multiple PDFs",
        page_icon=":books:",
        layout="wide",  # optional: set layout to wide for more space
    )
    st.title("Chat with multiple PDFs :books:")

    # Sidebar for Navigation
    with st.sidebar:
        st.header("Navigation")
        selection = st.radio(
            "Go to",
            ("Upload & Process PDFs", "Chat with PDFs", "View Chat History"),
            index=0,
        )

    # INITIALIZING SESSION STATE VARIABLES
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "display_chat_history" not in st.session_state:
        st.session_state.display_chat_history = []

    if "raw_text" not in st.session_state:
        st.session_state["raw_text"] = ""

    # Conditional display based on sidebar selection
    if selection == "Upload & Process PDFs":
        # file uploader to allow users to upload files
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'",
            accept_multiple_files=True,
            disabled=False,
        )
        if not pdf_docs:
            st.info("Kindly upload a pdf to chat :)")
        if st.button("Process"):
            if pdf_docs:
                with st.spinner("Kindly wait while we process your documents"):
                    # get pdf text
                    st.session_state.raw_text = get_pdf_text(pdf_docs)
                    # get the text chunks
                    text_chunks = get_text_chunks(st.session_state.raw_text)
                    # create vector store with embeddings
                    vectorstore = get_vectorstore(text_chunks)
                    # create conversation chain
                    st.session_state.rag_chain = get_conversation_chain(vectorstore)
            else:
                st.toast("Kindly enter a pdf", icon="⚠️")

    elif selection == "Chat with PDFs":
        if st.session_state.raw_text != "" and st.session_state.rag_chain:
            user_input = st.chat_input("Ask a question about your documents:")
            if user_input:
                # Display the user input
                with st.chat_message("user"):
                    st.markdown(user_input)
                # Append user input to display chat history
                st.session_state.display_chat_history.append(
                    {"role": "user", "content": user_input}
                )
                try:
                    # Generate response
                    response = st.session_state.rag_chain.invoke(
                        {
                            "input": user_input,
                            "chat_history": st.session_state.chat_history,
                        }
                    )
                    # Display response to user
                    with st.chat_message("assistant"):
                        st.markdown(response["answer"])
                    # Append response to display chat history
                    st.session_state.display_chat_history.append(
                        {"role": "assistant", "content": response["answer"]}
                    )
                    # Appending the user and bot responses to the chat history for the model
                    st.session_state.chat_history.extend(
                        [
                            HumanMessage(content=user_input),
                            AIMessage(content=response["answer"]),
                        ]
                    )
                except:
                    st.error("You have exhausted your API limits!!")
        else:
            st.info("Please upload and process PDFs first!")

    elif selection == "View Chat History":
        st.header("Chat History")
        if st.session_state.display_chat_history:
            for chat in st.session_state.display_chat_history:
                with st.chat_message(chat["role"]):
                    st.markdown(chat["content"])
        else:
            st.info("No chat history available yet!")


if __name__ == "__main__":
    main()
