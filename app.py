from llama_index.readers.azstorage_blob import AzStorageBlobReader
from azure.storage.blob import BlobServiceClient,generate_account_sas, ResourceTypes, AccountSasPermissions

from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core import VectorStoreIndex,StorageContext,Settings
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.postprocessor import SentenceTransformerRerank
import streamlit as st
import logging
import sys
import nest_asyncio
nest_asyncio.apply()
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

api_key = "c09f91126e51468d88f57cb83a63ee36"
azure_endpoint = "https://chat-gpt-a1.openai.azure.com/"
api_version = "2023-03-15-preview"


connect_str = 'DefaultEndpointsProtocol=https;AccountName=shreemirrahrag;AccountKey=UlbcKqbBtcs0J+2esmr8AznpwdV1dF5XI+v13kY07sjJ2U8rHmWbtCUqLjb12lD7Bn9k17mWgqjd+AStB4xXcw==;EndpointSuffix=core.windows.net'
container_name = 'ragfiles'
account_url='https://shreemirrahrag.blob.core.windows.net/ragfiles'
blob_service_client = BlobServiceClient(account_url=account_url, credentials='UlbcKqbBtcs0J+2esmr8AznpwdV1dF5XI+v13kY07sjJ2U8rHmWbtCUqLjb12lD7Bn9k17mWgqjd+AStB4xXcw==')

llm = AzureOpenAI(
    model="gpt-35-turbo-16k",
    deployment_name="DanielChatGPT16k",
    api_key=api_key,
    azure_endpoint=azure_endpoint,
    api_version=api_version,
)

embed_model = AzureOpenAIEmbedding(
    model="text-embedding-3-small",
    deployment_name="text-embedding-3-small",
    api_key="c09f91126e51468d88f57cb83a63ee36",
    azure_endpoint="https://chat-gpt-a1.openai.azure.com/",
    api_version="2023-03-15-preview",
)

Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 1000

vector_store = PGVectorStore.from_params(
    database='citus',
    host='c-shreemirrah-final.6c2zuyv2zdqcrp.postgres.cosmos.azure.com',
    password='Admin123!',
    port=5432,
    user='citus',
    table_name="vector_db_azure",
    embed_dim=1536)

index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
rerank = SentenceTransformerRerank(top_n = 2,model = "BAAI/bge-reranker-base")

query_engine = index.as_query_engine(similarity_top_k = 6, alpha=0.5,node_postprocessors = [postproc, rerank],)

def return_response(question):
    response = query_engine.query(question)
    answer=''
    response_text=response.response
    answer+=response_text+'\n\n'+'Source:\n\n'
    source=response.metadata

    for i in source.keys():
        answer+='\nSource:\n'
        answer+=source[i]['document_title']+'\n'
        answer+='\nFile-Name:\n'
        answer+=source[i]['file_name']+'\n'
        file=source[i]['file_name']
        blob_client = blob_service_client.get_blob_client(container_name, file)
        blob_url = blob_client.url
        answer+='\nFile-Path:\n'
        answer+=blob_url+'\n'
    return answer



st.set_page_config(page_title='RAG Questions Answered Extractor')
st.header('RAG Questions Answered Extractor')
input=st.text_input('Input: ',key="input")
submit=st.button("Ask")

if submit:
    st.subheader("Implementing...")
    st.write(return_response(input))