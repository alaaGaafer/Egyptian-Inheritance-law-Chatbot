from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
import os
from langchain.llms import OpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.llms import OpenAI
from openai import Client, OpenAI
import streamlit as st
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage
from langchain.chains import create_history_aware_retriever
import nltk
from nltk.tokenize import sent_tokenize
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import PyPDF2
import requests
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain import hub


OPENAI_API_KEY='sk-proj-coMvCzvEOcz9WsluNvoBT3BlbkFJPoIEq5AR2NX5LgpNLpIx'
os.environ["OPENAI_API_KEY"] = "sk-proj-coMvCzvEOcz9WsluNvoBT3BlbkFJPoIEq5AR2NX5LgpNLpIx"
client=OpenAI()

# RAG
class OnlinePDFLoader:
    def __init__(self, url):
        self.url = url

    def load_and_split(self):
        response = requests.get(self.url)
        with open("temp_pdf.pdf", "wb") as f:
            f.write(response.content)

        pages = []
        with open("temp_pdf.pdf", "rb") as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                pages.append(page.extract_text())
        return pages

loader = OnlinePDFLoader("https://azhar.gov.eg/materials/sec/sec3/%D8%A7%D9%84%D9%88%D8%AC%D9%8A%D8%B2%20%D9%81%D9%89%20%D8%A7%D9%84%D9%85%D9%8A%D8%B1%D8%A7%D8%AB.pdf")
pages = loader.load_and_split()

class TokenTextSplitter:
    def __init__(self, model_name="gpt-4", chunk_size=100, chunk_overlap=0):
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def create_documents(self, docs_text, metadatas=None):
        texts = []
        for idx, doc_text in enumerate(docs_text):
            sentences = sent_tokenize(doc_text)
            chunks = [sentences[i:i+self.chunk_size] for i in range(0, len(sentences), self.chunk_size - self.chunk_overlap)]
            for chunk_idx, chunk in enumerate(chunks):
                text = " ".join(chunk)
                metadata = metadatas[idx] if metadatas and len(metadatas) > idx else {}
                metadata["chunk_idx"] = chunk_idx
                texts.append({"document": text, "metadata": metadata})
        return texts

docs_text = pages[3:103]
doc_metadata = [{"document": "Inheritance law book"}]
tokenizer = TokenTextSplitter(model_name="gpt-4", chunk_size=100, chunk_overlap=0)
texts = tokenizer.create_documents(docs_text, metadatas=doc_metadata)
embeddings_model = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
docs_chunks = [chunk["document"] for chunk in texts]
docs_embeddings = embeddings_model.embed_documents(docs_chunks)
db_dir= "/content/Chroma_DB"

class Document:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata

texts = [Document(page_content=chunk["document"], metadata=chunk["metadata"]) for chunk in texts]
chroma_db = Chroma.from_documents(texts, embeddings_model, persist_directory=db_dir)

def format_output(output):
    if isinstance(output, list):
        output = output[0].page_content
    paragraphs = output.split("\n")
    formatted_output = ""
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if paragraph:
            formatted_output += f"{paragraph}\n\n"
    return formatted_output


llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=.5)
prompt = hub.pull("rlm/rag-prompt")

retriever = chroma_db.as_retriever(search_type="similarity", search_kwargs={"k": 6})
rag_chain = (
    {"context": retriever | format_output, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


# Chain Rag
contextualize_q_system_prompt = "Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."
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

qa_system_prompt = "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise. {context}"
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
chat_history = []

# Chat bot
def ask_question(chat_history, question):
  ai_msg = rag_chain.invoke({"input": question, "chat_history": chat_history})
  chat_history.extend([HumanMessage(content=question), ai_msg["answer"]])
  return ai_msg["answer"]

#while True:
#    question = input("Question: ")
#    if question.lower() == 'exit':
#        print("Exiting conversation.")
#        break
#    responce = ask_question(chat_history,question)
#    print(responce)

# UI --------------------------------------------------------------------------------------------------

st.title("ChatMwareeth")

with st.chat_message("assistant"):
    st.write("👋 أهلا! هل لديك سؤال يخص علم المواريث؟")

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Say something"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

#    with st.chat_message("assistant"):
#        stream = client.chat.completions.create(
#            model=st.session_state["openai_model"],
#            messages=[
#                {"role": m["role"], "content": m["content"]}
#                for m in st.session_state.messages
#            ],
#            stream=True,
#        )
#        response = st.write_stream(stream)
#        response = ask_question(chat_history, prompt)


    with st.chat_message("assistant"):
        ai_msg = rag_chain.invoke({"input": prompt, "chat_history": chat_history})
        #chat_history.extend([HumanMessage(content=prompt), ai_msg["answer"]])
        chat_history.extend(ai_msg["answer"])

        response = st.write_stream(chat_history)
    st.session_state.messages.append({"role": "assistant", "content": response})

