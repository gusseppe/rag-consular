# Import necessary modules
import streamlit as st
import hmac
# from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda, RunnableMap
from operator import itemgetter
from typing import List, Tuple
from langchain.schema import format_document
from langchain.prompts.prompt import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from glob import glob
# Load environment variables
# load_dotenv()

# Load documents and create the retriever
# paths = [
#     "./information/DNI_tramite.txt",
#     "./information/pasaporte_tramite.txt",
#     "./information/preguntas.txt",
#     "./atencion_sabados.txt",
#     ""
# ]
def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password.
        else:
            st.session_state["password_correct"] = False

    # Return True if the passward is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show input for password.
    st.text_input(
        "Password", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
    return False


if not check_password():
    st.stop()  # Do not continue if check_password is not True.

paths = glob("./information/*.txt")

loaders = [TextLoader(file_path=path, encoding='utf8') for path in paths]
documents = [doc for loader in loaders for doc in loader.load()]

text_splitter = RecursiveCharacterTextSplitter(chunk_size=6000, chunk_overlap=400) #4000. 200
documents = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(documents, embeddings, persist_directory="db")
retriever = vectorstore.as_retriever()

# Setup Memory and Chatbot Prompt
# memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
memory = ConversationBufferMemory(
    return_messages=True, output_key="answer", input_key="question"
)

loaded_memory = RunnablePassthrough.assign(
    chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("history"),
)

ANSWER_PROMPT = ChatPromptTemplate.from_template(
    template="""Eres un asistente consular de Peru para el consulado de Atlanta. Tu nombre es BravoAgent. Tu funcion es ayudar a responder preguntas del consulado de Atlanta, USA. 
    Usa tu base de conocimiento para responder preguntas de usuarios. 
    Responde la pregunta basándote únicamente en el siguiente contexto: {context}
    Pregunta: {question}"""
)

_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")


def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)

def _format_chat_history(chat_history: List[Tuple]) -> str:
    buffer = ""
    for dialogue_turn in chat_history:
        human = "Human: " + dialogue_turn[0]
        ai = "Assistant: " + dialogue_turn[1]
        buffer += "\n" + "\n".join([human, ai])
    return buffer


# Now we calculate the standalone question
standalone_question = {
    "standalone_question": {
        "question": lambda x: x["question"],
        "chat_history": lambda x: _format_chat_history(x["chat_history"]),
    }
    | CONDENSE_QUESTION_PROMPT
    | ChatOpenAI(temperature=0)
    | StrOutputParser(),
}
# Now we retrieve the documents
retrieved_documents = {
    "docs": itemgetter("standalone_question") | retriever,
    "question": lambda x: x["standalone_question"],
}
# Now we construct the inputs for the final prompt
final_inputs = {
    "context": lambda x: _combine_documents(x["docs"]),
    "question": itemgetter("question"),
}
# And finally, we do the part that returns the answers
answer = {
    "answer": final_inputs | ANSWER_PROMPT | ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0),
    "docs": itemgetter("docs"),
}
# And now we put it all together!
conversational_qa_chain = loaded_memory | standalone_question | retrieved_documents | answer


# Streamlit UI Setup
st.title("Asistente Consular")
st.markdown("Hola soy tu Agente Consular 🤓. Hazme una pregunta 🙋‍♂️")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Escribe tu pregunta aquí"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = conversational_qa_chain.invoke({"question": prompt, "chat_history": st.session_state.messages})
        message_placeholder.markdown(full_response["answer"].content)
    st.session_state.messages.append({"role": "assistant", "content": full_response["answer"].content})

    # Save memory after each interaction
    memory.save_context({"question": prompt}, {"answer": full_response["answer"].content})
