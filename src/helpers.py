import gc
import time

import torch

from utils.document_loaders import DocumentLoader
from utils.text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from utils.vectorstores import VectorStore
from utils.messages import AIMessage, HumanMessage, SystemMessage
from globals import embedding, retriever, prompt_template, llm, messages


def clear_cuda_cache():
    torch.cuda.empty_cache()
    gc.collect()


def index_files(files):
    global retriever

    docs = []
    for file in files:
        loader = DocumentLoader(file)
        docs.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
    splits = text_splitter.split_documents(docs)

    vectorstore = VectorStore.from_documents(documents=splits, embedding=embedding)

    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 10, "score_threshold": 0.55},
    )

    return "Finished!"


def user(message, history):
    return "", history + [[message, None]]


def bot(history):
    message = history[-1][0]
    retrieved_docs = retriever.get_relevant_documents(message)
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    augmented_prompt = prompt_template.format(context=context, query=message)

    messages.append(HumanMessage(content=augmented_prompt))
    response = llm.generate(messages)
    messages.append(response)

    history[-1][1] = ""
    for character in response.content:
        history[-1][1] += character
        time.sleep(0.01)
        yield history
