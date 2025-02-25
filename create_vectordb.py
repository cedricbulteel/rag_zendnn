import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import time

web_paths = ("https://lilianweng.github.io/posts/2023-06-23-agent/",
            "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
            "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
            "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
            "https://lilianweng.github.io/posts/2021-09-25-train-large/")


# Only keep post title, headers, and content from the full HTML.
bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
loader = WebBaseLoader(
    web_paths=web_paths,
    bs_kwargs={"parse_only": bs4_strainer},
)


docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

model_name = "BAAI/bge-m3"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}

embedding_model = HuggingFaceEmbeddings(model_name=model_name,
                                        model_kwargs=model_kwargs,
                                        encode_kwargs=encode_kwargs)

print('Embedding docs...')

collection_name = f"collection_of_webpages"
vectorstore = Chroma.from_documents(documents=all_splits, embedding=embedding_model, collection_name=collection_name, persist_directory="./vectordb")