from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.chains import (
    ConversationalRetrievalChain,
)

import os

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=100)
embeddings = OpenAIEmbeddings()

app = Flask(__name__)

@app.route("/message", methods=["GET", "POST"])
def reply_whatsapp():

    # try:
    #     num_media = int(request.values.get("NumMedia"))
    # except (ValueError, TypeError):
    #     return "Invalid request: invalid or missing NumMedia parameter", 400
    # response = MessagingResponse()
    # if not num_media:
    #     msg = response.message("Send us an image!")
    # else:
    #     msg = response.message("Thanks for the image. Here's one for you!")
    #     msg.media(GOOD_BOY_URL)
    # return str(response)

    file_uri = request.values.get("MediaUrl0")
    
    response = MessagingResponse()
    if not file_uri:
        if not os.path.isdir("chroma"):
            msg = response.message("No context provided. Please send a PDF file to query.")
        else:
            query = request.values.get('Body')
            vectordb = Chroma(persist_directory='chroma', embedding_function=embeddings)
            retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 1})

            message_history = ChatMessageHistory()

            memory = ConversationBufferMemory(
                memory_key="chat_history",
                output_key="answer",
                chat_memory=message_history,
                return_messages=True,
            )

            chain = ConversationalRetrievalChain.from_llm(
                ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, streaming=True),
                chain_type="stuff",
                retriever=retriever,
                memory=memory,
                return_source_documents=True,
            )

            result = chain.invoke(query)["answer"]

            msg = response.message(result)
    else:
        if request.values.get("MediaContentType0") == "application/pdf":
            loader = PyPDFLoader(file_uri)
            pages = loader.load_and_split()
            db = Chroma.from_documents(pages, OpenAIEmbeddings(), persist_directory='chroma')
            msg = response.message("PDF file recieved.")
        else:
            msg = response.message("Unsupported file type. Please send a PDF file.")
        
    return str(response)


if __name__ == "__main__":
    app.run()