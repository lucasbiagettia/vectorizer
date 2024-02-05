import os
import streamlit as st
from app_manager import AppManager
from src.inference_model_manager import InferenceModel
from src.embedding_model_manager import EmbeddingModel, EmbeddingModel

def initialize_session_variables():
    if "app_manager" not in st.session_state:
        st.session_state.app_manager = AppManager()

    if "db_name" not in st.session_state:
        st.session_state.db_name = 'vectorpoc'

    if "inference_model" not in st.session_state:
        st.session_state.inference_model = None

    if "embedding_model" not in st.session_state:
        st.session_state.embedding_model = None

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
    if st.session_state.inference_model is None:
        st.session_state.inference_model = InferenceModel()
    with st.chat_message("assistant"):
        db_name = st.session_state.db_name
        document = st.session_state.selected_document
        question = prompt
        embedding_model = st.session_state.embedding_model
        inference_model = st.session_state.inference_model
        app_manager = st.session_state.app_manager
        response = app_manager.make_question(db_name, document, question, embedding_model, inference_model)
        st.markdown(str(response))

def add_document(uploaded_file):
    app_manager = st.session_state.app_manager
    db_name = st.session_state.db_name
    app_manager.add_document(db_name, uploaded_file, st.session_state.embedding_model)

def get_available_documents():
    app_manager = st.session_state.app_manager
    db_name = st.session_state.db_name
    return app_manager.get_all_tables(db_name)

def print_all_messages(messages):
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def display_header():
    st.header("Vas a preguntar")

def handle_sidebar():
    display_inference_model_selection()
    display_embedding_model_selection()
    display_splitter_selection()
    display_document_selection()
    handle_file_upload()


def display_inference_model_selection():
    selected_model = st.text_input("Select a model", value="tiiuae/falcon-7b")
    st.write('Modelo seleccionado', selected_model)
    if selected_model is not None:
        st.session_state.inference_model = InferenceModel(model_id=selected_model)

def display_embedding_model_selection():
    embedding_models = ['PlanTL-GOB-ES/roberta-base-bne', 'sentence-transformers/LaBSE']
    embedding_model = st.selectbox('Select an option:', embedding_models)
    st.write('You have selected:', embedding_model)
    if embedding_model is not None:
        st.session_state.embedding_model = EmbeddingModel('sentence-transformers/LaBSE')

def display_splitter_selection():
    splitter_models = ['lbiagetti splitter']
    splitter_model = st.selectbox('Select an option:', splitter_models)
    st.write('You have selected:', splitter_model)
    if splitter_model is not None:
        st.session_state.splitter_model = splitter_model


def display_document_selection():
    options = get_available_documents()
    selected_option = st.selectbox('Select an option:', options)
    st.write('You have selected:', selected_option)
    if selected_option is not None:
        st.session_state.selected_document = selected_option

def handle_file_upload():
    uploaded_file = st.file_uploader('Select a file', type=['pdf'])
    file_name2 = st.text_input("Ingresa el nombre del archivo:")
    if st.button("Upload File"):
        process_file_upload(uploaded_file, file_name2)

def process_file_upload(uploaded_file, file_name):
    if uploaded_file is not None and file_name:
        add_document(uploaded_file)
        file_name, file_extension = os.path.splitext(uploaded_file.name)
        st.write(f'You have uploaded the file: {file_name}')
    else:
        st.write('Debes seleccionar un archivo y asignarle nombre')

def display_chat_messages():
    print_all_messages(st.session_state.messages)

def main():
    initialize_session_variables()
    display_header()
    with st.sidebar:
        handle_sidebar()
    display_chat_messages()
    prompt = st.chat_input("Ayúdame a continuar escribiendo")
    if prompt:
        handle_chat_input(prompt)

if __name__ == "__main__":
    main()


