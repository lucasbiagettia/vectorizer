import streamlit as st
from src.config.config_provider import ConfigProvider
from app_manager import AppManager
from src.qa_chain import QuestionAnsweringChain
from src.embedding_model_manager import EmbeddingModel

config_provider = ConfigProvider()

def initialize_session_variables():
    if "app_manager" not in st.session_state:
        st.session_state.app_manager = AppManager('vectorpoc')

    if "inference_model_name" not in st.session_state:
        st.session_state.inference_model = None
    
    if "embedding_model_name" not in st.session_state:
        st.session_state.embedding_model_name = None

    if "selected_document" not in st.session_state:
        st.session_state.selected_document = None

    if "splitter_model" not in st.session_state:
        st.session_state.splitter_model = None

    if "messages" not in st.session_state:
        st.session_state.messages = []


initialize_session_variables()


def handle_chat_input(prompt):
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        document = st.session_state.selected_document
        question = prompt
        embedding_model_name = st.session_state.embedding_model_name
        if st.session_state.inference_model_name is None:
            st.session_state.inference_model_name = config_provider.get_default_inference_model()
        inference_model_name = st.session_state.inference_model_name
        app_manager = st.session_state.app_manager
        response = app_manager.make_question(document, question, embedding_model_name, inference_model_name)
        st.markdown(str(response))

def add_document(uploaded_file):
    app_manager = st.session_state.app_manager
    app_manager.add_document(uploaded_file, st.session_state.embedding_model)

def get_available_documents():
    app_manager = st.session_state.app_manager
    return app_manager.get_all_documents()


def display_header():
    st.header("Vas a preguntar")

def handle_sidebar():
    select_model_and_document()

    upload_document()

def display_inference_model_selection():
    selected_model = st.text_input("Select a model", value= config_provider.get_default_inference_model())
    st.write('Modelo seleccionado', selected_model)
    return selected_model

def display_embedding_model_selection():
    embedding_model = st.text_input('Select an embedding model:', value = config_provider.get_default_embedding_model())
    return embedding_model


def display_document_selection():
    options = get_available_documents()
    selected_document = st.selectbox('Select an option:', options)
    st.write('You have selected:', selected_document)
    return selected_document


def select_model_and_document():
    #if st.button("Upload Document"):
        with st.form("Select model and document"):
            st.subheader('Inference model')
            inference_model_name = display_inference_model_selection()
            st.subheader('Document')
            document = display_document_selection()


            if st.form_submit_button("Confirm selection"):
                if "inference_model_name" not in st.session_state or st.session_state.inference_model_name is None:
                    st.session_state.inference_model_name = inference_model_name
                
                if "selected_document" not in st.session_state or st.session_state.selected_document is None:
                    st.session_state.selected_document = document
                    st.session_state.embedding_model_name = document[2]



def display_splitter_selection():
    splitter_models = ['lbiagetti splitter']
    splitter_model = st.selectbox('Select an option:', splitter_models)
    st.write('You have selected:', splitter_model)
    return splitter_model

def file_upload_and_input():
    uploaded_file = st.file_uploader('Select a file', type=['pdf'])
    return uploaded_file

def print_in_main():
    st.markdown("## Chat History")

        
def upload_document():
    #if st.button("Upload Document"):
        with st.form("Upload file"):
            st.subheader('Embedding model')
            embedding_model_name = display_embedding_model_selection()
            st.subheader('Splitter model')
            splitter = display_splitter_selection()
            st.subheader('Upload file')
            uploaded_file = file_upload_and_input()

            if st.form_submit_button("Confirm"):
                st.markdown("## Chat History")
                if "embedding_model_name" not in st.session_state or st.session_state.embedding_model_name is None:
                    st.session_state.embedding_model_name = embedding_model_name

                if "selected_document" not in st.session_state or st.session_state.selected_document is None:
                    st.session_state.selected_document = uploaded_file
                        
                if "splitter_model" not in st.session_state or st.session_state.splitter_model is None:
                    st.session_state.splitter_model = splitter
                
                process_file_upload()


def process_file_upload():
    file = st.session_state.selected_document
    model_name = st.session_state.embedding_model_name
    splitter = st.session_state.splitter_model
    app_manager = st.session_state.app_manager
    app_manager.add_document(file, model_name, splitter)
   

# def display_chat_messages():
#     print_all_messages(st.session_state.messages)

def main():
    initialize_session_variables()
    display_header()
    with st.sidebar:
        handle_sidebar()
    # display_chat_messages()
    prompt = st.chat_input("Ayúdame a continuar escribiendo")
    if prompt:
        handle_chat_input(prompt)

if __name__ == "__main__":
    main()


