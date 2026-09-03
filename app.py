import streamlit as st
from rag.embeddings import EmbeddingManager
from rag.vector_store import VectorStore
from rag.retriever import RAGRetriever
from rag.pipeline import create_llm, advanced_rag
import threading
import time

st.set_page_config(page_title="Universal RAG Assistant", page_icon="V", layout="centered")

st.title("UNIVERSAL RAG ASSISTANT")
st.write("""
                Ask questions about the loaded documents and get answers based on the documents    
                    stored in the knowledge base.
""")

# Load resources


@st.cache_resource
def load_embedding_manager():
    return EmbeddingManager()
@st.cache_resource
def load_vector_store():
    return VectorStore()
@st.cache_resource
def load_llm():
    return create_llm()
@st.cache_resource
def load_retriever():
    embedding_manager = load_embedding_manager()
    vector_store = load_vector_store()
    return RAGRetriever(vector_store, embedding_manager)


# Initialize components

try:
    embedding_manager = load_embedding_manager()
    vector_store = load_vector_store()
    retriever = load_retriever()
    llm = load_llm()

except Exception as e:
    st.error(f"Error initializing application: {e}")
    st.stop()

# Check database

document_count = (vector_store.collection.count())
st.sidebar.header("RAG Settings")
st.sidebar.write(
    f"Documents in database: "
    f"**{document_count}**"
)
top_k = st.sidebar.slider(
    "Number of retrieved chunks",
    min_value=1,
    max_value=10,
    value=3
)
score_threshold = st.sidebar.slider(
    "Similarity threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.2,
    step=0.05
)
# User query

query = st.text_input("Ask a question:", placeholder="e.g. Summarize the main points of the documents.")
if st.button("Ask", type="primary"):
    if not query.strip():
        st.warning("Please enter a question.")

    elif document_count == 0:
        st.warning(
            "The vector database is empty. "
            "Run `python ingest.py` first."
        )
    else:

        def show_loading_bar():
            st.markdown(
                """
                <style>
                .loading-container {
                    width: 100%;
                    height: 8px;
                    background-color: #262730;
                    border-radius: 10px;
                    overflow: hidden;
                    margin: 20px 0;
                }

                .loading-bar {
                    height: 100%;
                    width: 30%;
                    background: linear-gradient(
                        90deg,
                        #ff4b4b,
                        #ff6b6b,
                        #ff4b4b
                    );
                    border-radius: 10px;
                    animation: loading 1.8s ease-in-out infinite;
                }

                @keyframes loading {
                    0% {
                        transform: translateX(-120%);
                    }
                    50% {
                        transform: translateX(250%);
                    }
                    100% {
                        transform: translateX(450%);
                    }
                }
                </style>

                <div class="loading-container">
                    <div class="loading-bar"></div>
                </div>
                """,
                unsafe_allow_html=True
            )

        loading_placeholder = st.empty()

        with loading_placeholder.container():

            st.write("🤖 Searching documents and generating answer...")

            show_loading_bar()

        try:

            result = advanced_rag(
                query=query,
                retriever=retriever,
                llm=llm,
                top_k=top_k,
                score_threshold=score_threshold,
                return_context=True
            )

            loading_placeholder.empty()

            st.subheader("Answer")
            st.write(result["answer"])

            st.caption(
                "Best retrieval similarity: "
                f"{result['retrieval_score']:.3f}"
            )


            st.subheader("Sources")

            for i, source in enumerate(
                result["source"],
                start=1
            ):
                with st.expander(
                    f"Source {i}: {source['source']}"
                ):
                    st.write(
                        f"*Page:* {source['page']}"
                    )

                    st.write(
                        f"*Similarity:* "
                        f"{source['score']:.3f}"
                    )

                    st.write(
                        source["preview"]
                    )

            with st.expander(
                "View retrieved context"
            ):
                st.write(
                    result["context"]
                )

        except Exception as e:
            
            loading_placeholder.empty()
            st.error(
                f"An error occurred: {e}"
            )