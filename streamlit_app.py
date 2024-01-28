from inference_model.answer_generator import ConversationalAgent
import streamlit as st
from vectorizer import EmbeddingModel
from db_manager import PosgresManager



if "chatbot" not in st.session_state:
    st.session_state.chatbot = ConversationalAgent()

if "db_manager" not in st.session_state:
    st.session_state.db_manager = PosgresManager('vectorpoc')
    st.session_state.db_manager.connect()

if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = EmbeddingModel()
def reset_values():
    st.session_state.messages = []
    st.session_state.chatbot.clean()

def finalize():
    st.session_state.messages = []
    assistant_text = "Mira lo que hemos escrito: \n\n"
    final_generated_text = st.session_state.chatbot.get_total_text()
    st.session_state.chatbot.clean()
    st.session_state.messages.append({"role": "assistant", "content": assistant_text + final_generated_text})


def handle_chat_input(prompt):




    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        embedding = st.session_state.embedding_model.get_embedding(prompt)
        sim_docs = st.session_state.db_manager.get_similar_docs(embedding, 5, 'marx')
        response = st.session_state.chatbot.answer_question(prompt, sim_docs)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})



def print_all_messages(messages):
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def main():
    st.header("Vas a preguntar")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    with st.sidebar:
        if st.sidebar.button("Finalizar"):
            finalize()
        if st.sidebar.button("Reiniciar"):
            reset_values()

        st.subheader("Estemos en contacto:")

        # linkedin_logo = "social_logos/LI-In-Bug.png"
        # st.image(linkedin_logo, width=16, use_column_width=False)
        # st.markdown(f"[@lucasbiagettia](https://www.linkedin.com/in/lucasbiagettia/)")

        # github_logo = "social_logos/github-mark.png"
        # st.image(github_logo, width=16, use_column_width=False)
        # st.markdown(f"[@lucasbiagettia](https://github.com/lucasbiagettia/)")

        # hugging_face_logo = "social_logos/hf-logo.png"
        # st.image(hugging_face_logo, width=16, use_column_width=False)
        # st.markdown(f"[@lucasbiagettia](https://huggingface.co/lucasbiagettia)")


  

    print_all_messages(st.session_state.messages)

    prompt = st.chat_input("Ayúdame a continuar escribiendo")
    if prompt:
        handle_chat_input(prompt)


if __name__ == "__main__":
    main()