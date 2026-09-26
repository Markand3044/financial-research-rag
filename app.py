import json
import uuid
from pathlib import Path

import requests
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================

API_URL = "http://127.0.0.1:8000"

CHAT_HISTORY_FILE = Path("data/chat_history.json")

CHAT_HISTORY_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Financial Research RAG",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main page */

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 5rem;
        max-width: 1100px;
    }


    /* Sidebar */

    section[data-testid="stSidebar"] {
        width: 300px !important;
    }


    /* New chat button */

    .new-chat-button {
        width: 100%;
    }


    /* Source box */

    .source-box {
        padding: 10px 14px;
        border-radius: 8px;
        border: 1px solid rgba(128, 128, 128, 0.25);
        margin-top: 8px;
        margin-bottom: 10px;
    }


    /* Document information */

    .document-info {
        padding: 10px;
        border-radius: 8px;
        border: 1px solid rgba(128, 128, 128, 0.25);
        margin-top: 8px;
    }


    /* Conversation title */

    .conversation-title {
        font-size: 14px;
    }


    /* Hide Streamlit decoration */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "conversations" not in st.session_state:
    st.session_state.conversations = {}


if "current_conversation_id" not in st.session_state:
    st.session_state.current_conversation_id = None


if "documents" not in st.session_state:
    st.session_state.documents = []


if "document_search" not in st.session_state:
    st.session_state.document_search = ""


# ============================================================
# CHAT HISTORY FUNCTIONS
# ============================================================

def load_chat_history():
    """
    Load conversations from local JSON storage.
    """

    if not CHAT_HISTORY_FILE.exists():
        return {}

    try:

        with CHAT_HISTORY_FILE.open(
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

        return data

    except Exception:

        return {}


def save_chat_history():
    """
    Save conversations to local JSON storage.
    """

    try:

        with CHAT_HISTORY_FILE.open(
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                st.session_state.conversations,
                file,
                indent=4,
                ensure_ascii=False
            )

    except Exception as e:

        st.error(
            f"Could not save chat history: {e}"
        )


# Load conversations only once
if not st.session_state.conversations:

    st.session_state.conversations = load_chat_history()


# ============================================================
# CONVERSATION FUNCTIONS
# ============================================================

def create_conversation(document_id=None):
    """
    Create a new conversation with a unique ID.
    """

    conversation_id = str(uuid.uuid4())

    st.session_state.conversations[conversation_id] = {
        "title": "New conversation",
        "document_id": document_id,
        "messages": []
    }

    st.session_state.current_conversation_id = conversation_id

    save_chat_history()

    return conversation_id


def get_current_conversation():

    conversation_id = (
        st.session_state.current_conversation_id
    )

    if conversation_id is None:
        return None

    return st.session_state.conversations.get(
        conversation_id
    )


def delete_conversation(conversation_id):

    if conversation_id in st.session_state.conversations:

        del st.session_state.conversations[
            conversation_id
        ]

    if (
        st.session_state.current_conversation_id
        == conversation_id
    ):

        st.session_state.current_conversation_id = None

    save_chat_history()


def generate_conversation_title(question):

    """
    Create a simple title from the first user question.
    """

    title = question.strip()

    if len(title) > 45:

        title = title[:45].rstrip() + "..."

    return title


# ============================================================
# API FUNCTIONS
# ============================================================

def get_documents_from_api():

    try:

        response = requests.get(
            f"{API_URL}/documents",
            timeout=30
        )

        if response.status_code == 200:

            return response.json().get(
                "documents",
                []
            )

        return []

    except requests.exceptions.RequestException:

        return []


def upload_document_to_api(uploaded_file):

    try:

        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                "application/pdf"
            )
        }

        response = requests.post(
            f"{API_URL}/upload",
            files=files,
            timeout=600
        )

        return response

    except requests.exceptions.RequestException as e:

        return e


def delete_document_from_api(document_id):

    try:

        response = requests.delete(
            f"{API_URL}/documents/{document_id}",
            timeout=30
        )

        return response

    except requests.exceptions.RequestException as e:

        return e


def ask_rag(question, document_id):

    try:

        response = requests.post(
            f"{API_URL}/ask",
            json={
                "question": question,
                "document_id": document_id
            },
            timeout=600
        )

        return response

    except requests.exceptions.RequestException as e:

        return e


# ============================================================
# LOAD DOCUMENTS
# ============================================================

st.session_state.documents = get_documents_from_api()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("📊 Financial Research")

    # --------------------------------------------------------
    # NEW CHAT
    # --------------------------------------------------------

    if st.button(
        "＋ New Chat",
        use_container_width=True
    ):

        create_conversation()

        st.rerun()


    st.divider()


    # --------------------------------------------------------
    # DOCUMENTS
    # --------------------------------------------------------

    st.subheader("📁 Financial Reports")


    uploaded_file = st.file_uploader(
        "Upload a report",
        type=["pdf"],
        label_visibility="collapsed"
    )


    if uploaded_file is not None:

        if st.button(
            "Upload Report",
            use_container_width=True
        ):

            with st.spinner(
                "Processing financial report..."
            ):

                result = upload_document_to_api(
                    uploaded_file
                )


            if isinstance(
                result,
                requests.Response
            ):

                if result.status_code == 200:

                    st.success(
                        "Report uploaded successfully."
                    )

                    st.session_state.documents = (
                        get_documents_from_api()
                    )

                    st.rerun()

                else:

                    try:

                        detail = result.json().get(
                            "detail",
                            "Upload failed."
                        )

                    except Exception:

                        detail = "Upload failed."

                    st.error(detail)

            else:

                st.error(
                    f"Could not connect to API: {result}"
                )


    # --------------------------------------------------------
    # DOCUMENT SEARCH
    # --------------------------------------------------------

    st.text_input(
        "Search reports",
        key="document_search",
        placeholder="Search reports..."
    )


    search_text = (
        st.session_state.document_search
        .strip()
        .lower()
    )


    filtered_documents = [

        document

        for document in st.session_state.documents

        if (
            not search_text
            or search_text in document["filename"].lower()
        )

    ]


    # --------------------------------------------------------
    # DOCUMENT LIST
    # --------------------------------------------------------

    if not filtered_documents:

        st.caption(
            "No matching reports found."
        )

    else:

        for document in filtered_documents:

            document_id = document["document_id"]

            filename = document["filename"]


            # Use a container for each report

            with st.container(border=True):

                st.write(
                    f"📄 **{filename}**"
                )


                if st.button(
                    "Use this report",
                    key=f"use_{document_id}",
                    use_container_width=True
                ):

                    conversation = (
                        get_current_conversation()
                    )

                    if conversation is None:

                        conversation_id = (
                            create_conversation(
                                document_id
                            )
                        )

                    else:

                        conversation["document_id"] = (
                            document_id
                        )

                        save_chat_history()


                    st.rerun()


                if st.button(
                    "Delete",
                    key=f"delete_{document_id}",
                    use_container_width=True
                ):

                    response = (
                        delete_document_from_api(
                            document_id
                        )
                    )


                    if isinstance(
                        response,
                        requests.Response
                    ):

                        if response.status_code == 200:

                            st.success(
                                "Report deleted."
                            )

                            # Remove deleted report
                            # from current conversations

                            for conversation in (
                                st.session_state
                                .conversations
                                .values()
                            ):

                                if (
                                    conversation
                                    .get("document_id")
                                    == document_id
                                ):

                                    conversation[
                                        "document_id"
                                    ] = None


                            save_chat_history()

                            st.session_state.documents = (
                                get_documents_from_api()
                            )

                            st.rerun()

                        else:

                            try:

                                detail = (
                                    response
                                    .json()
                                    .get(
                                        "detail",
                                        "Delete failed."
                                    )
                                )

                            except Exception:

                                detail = "Delete failed."

                            st.error(detail)

                    else:

                        st.error(
                            f"Could not connect to API: {response}"
                        )


    st.divider()


    # --------------------------------------------------------
    # CONVERSATION HISTORY
    # --------------------------------------------------------

    st.subheader("💬 Conversations")


    if not st.session_state.conversations:

        st.caption(
            "No conversations yet."
        )


    else:

        for (
            conversation_id,
            conversation
        ) in reversed(
            list(
                st.session_state
                .conversations
                .items()
            )
        ):

            title = conversation.get(
                "title",
                "New conversation"
            )


            col1, col2 = st.columns(
                [5, 1]
            )


            with col1:

                if st.button(
                    title,
                    key=f"chat_{conversation_id}",
                    use_container_width=True
                ):

                    st.session_state.current_conversation_id = (
                        conversation_id
                    )

                    st.rerun()


            with col2:

                if st.button(
                    "⋮",
                    key=f"menu_{conversation_id}"
                ):

                    delete_conversation(
                        conversation_id
                    )

                    st.rerun()


# ============================================================
# CREATE INITIAL CONVERSATION
# ============================================================

conversation = get_current_conversation()


if conversation is None:

    if st.session_state.documents:

        first_document = (
            st.session_state.documents[0]
        )

        create_conversation(
            first_document["document_id"]
        )

    else:

        create_conversation()


    conversation = get_current_conversation()


# ============================================================
# MAIN HEADER
# ============================================================

st.title(
    "📊 Financial Research RAG Assistant"
)


st.caption(
    "Ask questions about your financial reports "
    "using retrieval-augmented generation."
)


# ============================================================
# CURRENT DOCUMENT
# ============================================================

selected_document_id = conversation.get(
    "document_id"
)


selected_document = None


for document in st.session_state.documents:

    if document["document_id"] == selected_document_id:

        selected_document = document

        break


if selected_document:

    st.info(
        f"📄 **Report:** "
        f"{selected_document['filename']}"
    )

else:

    if st.session_state.documents:

        st.warning(
            "Select a financial report from the sidebar "
            "before asking a question."
        )

    else:

        st.warning(
            "Upload a financial report from the sidebar "
            "to start your research."
        )


st.divider()


# ============================================================
# CHAT HISTORY
# ============================================================

for message in conversation["messages"]:

    role = message["role"]

    content = message["content"]


    with st.chat_message(role):

        st.markdown(content)


        # Show sources for assistant messages

        if (
            role == "assistant"
            and message.get("source_pages")
        ):

            with st.expander(
                "📚 Sources"
            ):

                for page in message[
                    "source_pages"
                ]:

                    st.write(
                        f"📄 PDF Page {page}"
                    )


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask a question about the financial report..."
)


# ============================================================
# ASK RAG
# ============================================================

if question:

    question = question.strip()


    if not question:

        st.stop()


    # --------------------------------------------------------
    # Check document
    # --------------------------------------------------------

    if not selected_document_id:

        st.warning(
            "Please select a financial report first."
        )

        st.stop()


    # --------------------------------------------------------
    # Add user message
    # --------------------------------------------------------

    conversation["messages"].append(
        {
            "role": "user",
            "content": question
        }
    )


    # --------------------------------------------------------
    # Generate title from first question
    # --------------------------------------------------------

    if (
        conversation["title"]
        == "New conversation"
    ):

        conversation["title"] = (
            generate_conversation_title(
                question
            )
        )


    save_chat_history()


    # --------------------------------------------------------
    # Display user question
    # --------------------------------------------------------

    with st.chat_message("user"):

        st.markdown(question)


    # --------------------------------------------------------
    # Call RAG
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "Researching the financial report..."
        ):

            response = ask_rag(
                question,
                selected_document_id
            )


        if isinstance(
            response,
            requests.Response
        ):

            if response.status_code == 200:

                result = response.json()

                answer = result.get(
                    "answer",
                    "No answer returned."
                )

                source_pages = result.get(
                    "source_pages",
                    []
                )

                citations = result.get(
                    "citations",
                    []
                )


                # ------------------------------------------------
                # Display answer
                # ------------------------------------------------

                st.markdown(answer)


                # ------------------------------------------------
                # Display sources
                # ------------------------------------------------

                if source_pages:

                    with st.expander(
                        "📚 Sources"
                    ):

                        for page in source_pages:

                            st.write(
                                f"📄 PDF Page {page}"
                            )


                # ------------------------------------------------
                # Save assistant message
                # ------------------------------------------------

                conversation["messages"].append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "source_pages": source_pages,
                        "citations": citations
                    }
                )


                save_chat_history()


            else:

                try:

                    detail = (
                        response
                        .json()
                        .get(
                            "detail",
                            "The RAG request failed."
                        )
                    )

                except Exception:

                    detail = (
                        "The RAG request failed."
                    )


                st.error(detail)


        else:

            st.error(
                f"Could not connect to FastAPI: {response}"
            )