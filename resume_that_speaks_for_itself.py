import chromadb
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import (Settings, VectorStoreIndex, SimpleDirectoryReader, PromptTemplate)
from llama_index.core import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import SentenceSplitter
import argparse

def main():
    #import LLM and embedding model
    llm = Ollama(model="llama3")
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.llm = llm
    Settings.embed_model = embed_model

    #read the document
    documents = SimpleDirectoryReader(input_files=["./Diana_Morales_Resume.pdf"]).load_data()

    #set up a vector database
    chroma_client = chromadb.EphemeralClient()
    chroma_collection = chroma_client.create_collection("ollama")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    #put it all together in an index
    index = VectorStoreIndex.from_documents(documents, 
                                            storage_context=storage_context, 
                                            embed_model=embed_model,
                                            transformations=[SentenceSplitter(chunk_size=256, chunk_overlap=10)])

    #custom prompt template
    template = (
        "Imagine you are a data scientit's assistant and you answer recruiter's questions about the data scientist's experience."
        "Here is some context from the data scientist's resume related to the query::\n"
        "-----------------------------------------\n"
        "{context_str}\n"
        "-----------------------------------------\n"
        "Considering the above information, please respond to the following inquiry:\n\n"
        "Question: {query_str}\n\n"
        "Answer succinctly and ensure your response is understandable to someone without data science background."
        "The data scientist's name is Diana."
    )
    qa_template = PromptTemplate(template)

    # build query engine with custom template
    query_engine = index.as_query_engine(text_qa_template=qa_template, similarity_top_k=3)

    #get the question from command line
    question = input("Resume Assistant: What would you like to know about Diana?\n User: ")

    #run the engine
    response = query_engine.query(question)
    print(response.response)
    return 1

if __name__ == "__main__":
    main()