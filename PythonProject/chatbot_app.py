import streamlit as st
from assistant_manager import AssistantManager


def main():
    st.set_page_config(page_title="Manager AI Assistant", page_icon="🤖")

    # Initialize assistant
    if "assistant" not in st.session_state:
        st.session_state.assistant = AssistantManager()

    st.title("🤖 Manager AI Assistant")
    st.markdown("I can help with data analysis, predictions, and general questions!")

    # Sample questions
    st.sidebar.markdown("**Try asking:**")
    st.sidebar.markdown("- Show me data analysis of Titanic dataset")
    st.sidebar.markdown("- Predict survival for a 25 year old female in 1st class")
    st.sidebar.markdown("- What's the model accuracy?")
    st.sidebar.markdown("- Hello, how are you?")

    # Chat interface
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    if user_input := st.chat_input("Ask me anything..."):
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Get assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.assistant.process_message(user_input)
                st.markdown(response)

        st.session_state.chat_history.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()