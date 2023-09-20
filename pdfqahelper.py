from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import os
import openai
import pandas as pd



load_dotenv()
API_KEY = os.environ.get("API_KEY")
def pdf_qa(text):
    pdfreader = PdfReader('Budget Circular 2022-2023.pdf')
    pages = list(pdfreader.pages)

    from typing_extensions import Concatenate
    raw_text =''
    for i,page in enumerate(pdfreader.pages):
        content = page.extract_text()
        if content:
            raw_text += content
    #print(raw_text)

    # We need to split the text using Character Text Split such that it sshould not increse token size
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 800,
        chunk_overlap  = 200,
        length_function = len,
    )
    texts = text_splitter.split_text(raw_text)
    print(len(texts))
    embeddings = OpenAIEmbeddings()
    document_search = FAISS.from_texts(texts, embeddings)

    from langchain.chains.question_answering import load_qa_chain
    from langchain.llms import OpenAI

    chain = load_qa_chain(OpenAI(), chain_type="stuff")
    # query = "who participats in budget making"
    query = text
    docs = document_search.similarity_search(query)
    response = chain.run(input_documents=docs, question=query)
    print(response)
    return response;

if __name__ == '__main__':
    text = '''
    Who participates in the budget creation
    '''
    answer = pdf_qa(text)

    print(answer)
