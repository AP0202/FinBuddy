# Import os to set API key
import os
# Import OpenAI as main LLM service
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
# Bring in streamlit for UI/app interface
import streamlit as st
from api_key import apikey

# Import PDF document loaders...there's other ones as well!
from langchain.document_loaders import PyPDFLoader
# Import chroma as the vector store 
from langchain.vectorstores import Chroma
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain
from streamlit_chat import message

from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)

# Import vector store stuff
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)


# Set APIkey for OpenAI Service
# Can sub this out for other LLM providers
os.environ['OPENAI_API_KEY'] = apikey


# Create instance of OpenAI LLM
llm = OpenAI(temperature=0.1, verbose=True)
embeddings = OpenAIEmbeddings()

# Create and load PDF Loader
loader = PyPDFLoader('cognitive_computing_data_set_new.pdf')
# Split pages from pdf 
pages = loader.load_and_split()
# Load documents into vector database aka ChromaDB
store = Chroma.from_documents(pages, embeddings, collection_name='cognitive_computing_data_set_new')

# Create vectorstore info object - metadata repo?
vectorstore_info = VectorStoreInfo(
    name="cognitive_computing_data_set_new",
    description="energy sector report as a pdf",
    vectorstore=store
)
# Convert the document store into a langchain toolkit
toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)

# Add the toolkit to an end-to-end LC
agent_executor = create_vectorstore_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True
)


st.title('💰🔗 FinBuddy')
# Create a text input box for the user

st.write("🤖 Welcome to the FinAI Assistant ChatBox!")
st.write("Got burning questions about investing? You're in the right place! Our chatbot is geared up to assist you with insights, advice, and solutions. Just type in your queries, and let's unravel the mysteries of your questions together! 🔍💻")

if 'responses' not in st.session_state:
    st.session_state['responses'] = ["How can I assist you?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

if 'buffer_memory' not in st.session_state:
    st.session_state.buffer_memory=ConversationBufferWindowMemory(k=3,return_messages=True)

system_msg_template = SystemMessagePromptTemplate.from_template(template="""Answer the question as truthfully as possible using the provided context, 
                                                                                    and if the answer is not contained within the text below, say 'I don't know'""")
human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")
prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True)

response_container = st.container()
textcontainer = st.container()

with textcontainer:
    query = st.text_input("Hello! How can I help you? ", key="input")
    if query:
        with st.spinner("thinking..."):
            response = agent_executor.run(query)
            st.session_state.requests.append(query)
            st.session_state.responses.append(response)

with response_container:
    if st.session_state['responses']:
        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i],key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True,key=str(i)+ '_user')

# If the user hits enter
# if prompt:
#     # Then pass the prompt to the LLM
#     response = agent_executor.run(prompt)
#     # ...and write it out to the screen
#     st.write(response)

#     # With a streamlit expander  
#     with st.expander('Document Similarity Search'):
#         # Find the relevant pages
#         search = store.similarity_search_with_score(prompt) 
#         # Write out the first 
#         st.write(search[0][0].page_content) 
