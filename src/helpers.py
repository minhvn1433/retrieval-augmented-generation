import gc
import time

import torch

from utils.document_loaders import DocumentLoader
from utils.text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from utils.vectorstores import VectorStore
from utils.messages import AIMessage, HumanMessage, SystemMessage
from globals import (
    embedding,
    retriever,
    prompt_template,
    query_template,
    llm,
    cross_encoder,
    messages,
)


def clear_cuda_cache():
    torch.cuda.empty_cache()
    gc.collect()


def reordering_texts(texts):
    texts.reverse()
    reordered_result = []
    for i, value in enumerate(texts):
        if i % 2 == 1:
            reordered_result.append(value)
        else:
            reordered_result.insert(0, value)
    return reordered_result


def index_files(files):
    global retriever

    docs = []
    for file in files:
        loader = DocumentLoader(file)
        docs.extend(loader.load())

    text_splitter = CharacterTextSplitter(chunk_size=256, chunk_overlap=128)
    splits = text_splitter.split_documents(docs)

    vectorstore = VectorStore.from_documents(documents=splits, embedding=embedding)

    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 10}
    )

    return "Finished!"


def user(message, history):
    return "", history + [[message, None]]


def bot(history):
    message = history[-1][0]
    question = query_template.format(question=message)
    queries = llm.generate([HumanMessage(content=question)]).content.split("\n\n")

    retrieved_docs = []
    for query in queries:
        retrieved_docs.extend(retriever.get_relevant_documents(query))
    unique_contents = list({doc.page_content for doc in retrieved_docs})

    scores = cross_encoder.predict([(message, content) for content in unique_contents])
    reranked_texts = [
        doc for _, doc in sorted(zip(scores, unique_contents), reverse=True)[:5]
    ]
    reordered_texts = reordering_texts(reranked_texts)

    context = "\n".join(reordered_texts)
    augmented_prompt = prompt_template.format(context=context, query=message)
    print(augmented_prompt)

    messages.append(HumanMessage(content=augmented_prompt))
    response = llm.generate(messages)
    messages.append(response)

    history[-1][1] = ""
    for character in response.content:
        history[-1][1] += character
        time.sleep(0.01)
        yield history
