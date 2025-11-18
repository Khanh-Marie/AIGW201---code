import streamlit as st
from assistant_manager import AssistantManager

def main():
    st.set_page_config(page_title="Manager AI Assistant")

    if "assistant" not in st.session_state:
        st.session_state.assistant = AssistantManager()

    st.title("Manager AI Assistant")
    st.markdown("Hello, I can help with data analysis, make predictions, and general questions")

    st.sidebar.markdown("**Try asking**")
    st.sidebar.markdown("- Hello, how are you ?")
    st.sidebar.markdown("- What's the model accuracy ?")
    st.sidebar.markdown("- Show me the data analysis of the Titanic dataset")
    st.sidebar.markdown("- Predict the survival rate of a 25 year old female in 1st class")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_input := st.chat_input("Ask me anything..."):
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.assistant.process_message(user_input)
                st.markdown(response)

        st.session_state.chat_history.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()