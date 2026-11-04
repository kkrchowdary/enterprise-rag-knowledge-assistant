import streamlit as st
import requests


# ==================================================
# 1. APPLICATION CONFIGURATION
# ==================================================

API_BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Enterprise RAG Knowledge Assistant",
    page_icon="📚",
    layout="wide"
)


# ==================================================
# 2. SESSION STATE
# ==================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_documents" not in st.session_state:
    st.session_state.uploaded_documents = []


# ==================================================
# 3. API FUNCTIONS
# ==================================================

def upload_pdf(uploaded_file):
    """Upload a PDF document to FastAPI."""

    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            "application/pdf"
        )
    }

    response = requests.post(
        f"{API_BASE_URL}/documents/upload",
        files=files,
        timeout=300
    )

    response.raise_for_status()

    return response.json()


def ask_rag(question, top_k=3):
    """Send a question to the RAG backend."""

    payload = {
        "question": question,
        "top_k": top_k
    }

    response = requests.post(
        f"{API_BASE_URL}/rag/ask",
        json=payload,
        timeout=300
    )

    response.raise_for_status()

    return response.json()


# ==================================================
# 4. ERROR HANDLING
# ==================================================

def get_api_error_message(error):
    """
    Extract a readable error message from FastAPI.
    """

    if isinstance(error, requests.exceptions.HTTPError):

        try:
            response_data = error.response.json()

            return response_data.get(
                "detail",
                str(error)
            )

        except (ValueError, AttributeError):
            return str(error)

    return str(error)


# ==================================================
# 5. SOURCE CITATIONS
# ==================================================

def display_sources(sources):
    """Display source citations and retrieved passages."""

    if not sources:
        return

    with st.expander(
        f"View Sources ({len(sources)})"
    ):

        for source in sources:

            citation_id = source.get(
                "citation_id",
                "?"
            )

            filename = source.get(
                "source",
                "Unknown"
            )

            page = source.get(
                "page",
                "Unknown"
            )

            distance = source.get(
                "distance",
                "N/A"
            )

            cited = source.get(
                "cited",
                False
            )

            content = source.get(
                "content",
                ""
            )

            st.markdown(
                f"### [{citation_id}] {filename}"
            )

            st.caption(
                f"Page: {page} | "
                f"Distance: {distance} | "
                f"Referenced in answer: {cited}"
            )

            st.markdown("**Retrieved Passage:**")

            st.info(content)

            st.divider()


# ==================================================
# 6. SIDEBAR
# ==================================================

with st.sidebar:

    st.title("📚 Enterprise RAG")

    st.markdown("### Upload Documents")

    uploaded_file = st.file_uploader(
        "Choose a PDF",
        type=["pdf"]
    )

    if uploaded_file is not None:

        if st.button(
            "Process Document",
            use_container_width=True
        ):

            with st.spinner("Processing PDF..."):

                try:

                    result = upload_pdf(uploaded_file)

                    document_info = {
                        "filename": result.get(
                            "filename",
                            uploaded_file.name
                        ),
                        "pages": result.get(
                            "pages",
                            0
                        ),
                        "chunks": result.get(
                            "chunks",
                            0
                        )
                    }

                    st.session_state.uploaded_documents.append(
                        document_info
                    )

                    st.success(
                        "Document processed successfully!"
                    )

                    st.write(
                        f"Pages: {document_info['pages']}"
                    )

                    st.write(
                        f"Chunks: {document_info['chunks']}"
                    )

                except requests.exceptions.HTTPError as error:

                    message = get_api_error_message(error)

                    st.error(
                        f"Upload failed: {message}"
                    )

                except requests.exceptions.RequestException as error:

                    st.error(
                        f"Backend connection failed: {error}"
                    )

                except Exception as error:

                    st.error(
                        f"Unexpected upload error: {error}"
                    )

    st.divider()

    # ----------------------------------------------
    # Uploaded documents
    # ----------------------------------------------

    st.markdown("### Uploaded Documents")

    if st.session_state.uploaded_documents:

        for document in st.session_state.uploaded_documents:

            st.write(
                f"📄 {document['filename']}"
            )

            st.caption(
                f"{document['pages']} pages | "
                f"{document['chunks']} chunks"
            )

    else:

        st.info(
            "No documents uploaded in this session."
        )

    st.divider()

    # ----------------------------------------------
    # Retrieval settings
    # ----------------------------------------------

    st.markdown("### Retrieval Settings")

    top_k = st.slider(
        "Number of retrieved chunks",
        min_value=1,
        max_value=10,
        value=3
    )

    st.divider()

    # ----------------------------------------------
    # Clear chat
    # ----------------------------------------------

    if st.button(
        "Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


# ==================================================
# 7. MAIN CHAT INTERFACE
# ==================================================

st.title("Enterprise RAG Knowledge Assistant")

st.caption(
    "Upload PDF documents and ask questions "
    "using Retrieval-Augmented Generation."
)

st.divider()


# ==================================================
# 8. DISPLAY CHAT HISTORY
# ==================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message["role"] == "assistant":

            display_sources(
                message.get(
                    "sources",
                    []
                )
            )


# ==================================================
# 9. USER QUESTION
# ==================================================

question = st.chat_input(
    "Ask a question about your uploaded documents..."
)

if question:

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):

        st.markdown(question)

    # ----------------------------------------------
    # Generate RAG answer
    # ----------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "Searching documents and generating answer..."
        ):

            try:

                result = ask_rag(
                    question=question,
                    top_k=top_k
                )

                answer = result.get(
                    "answer",
                    "No answer generated."
                )

                sources = result.get(
                    "sources",
                    []
                )

                st.markdown(answer)

                display_sources(sources)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

            except requests.exceptions.HTTPError as error:

                message = get_api_error_message(error)

                st.error(
                    f"RAG request failed: {message}"
                )

            except requests.exceptions.RequestException as error:

                st.error(
                    f"Backend connection failed: {error}"
                )

            except Exception as error:

                st.error(
                    f"Unexpected RAG error: {error}"
                )