# 📚 Universal PDF RAG Assistant

A modular Retrieval-Augmented Generation (RAG) application that allows users to upload or provide PDF documents and interact with their content using natural language.

Instead of searching through lengthy documents manually, users can ask questions and receive answers based on the information retrieved from the provided documents.

The project was built as a reusable RAG pipeline rather than a domain-specific chatbot, meaning the same system can be used with different collections of PDF documents.

## 🎥 Demo

<img width="480" height="248" alt="demo" src="https://github.com/user-attachments/assets/289d27ef-b920-4784-af65-102923bda1c4" />

## 🧠 What Does It Do?

Imagine you have:

- 📄 A 200-page textbook
- 📑 A collection of research papers
- 📋 Company documentation
- 📚 Lecture notes
- 📖 Technical manuals
- 🗂️ Multiple PDF reports

Instead of manually searching through all of them, you can ask:

> "What are the main conclusions of this report?"

or:

> "Explain the methodology used in this paper."

The application searches the provided documents for the most relevant information and gives that information to an LLM to generate a contextual answer.

### In simple terms:

**Your PDFs → Find relevant information → Give it to AI → Generate an answer**

## 🔍 Why RAG?

A traditional LLM generates an answer primarily from the knowledge it has learned during training.

This project takes a different approach.

Instead of asking the model to answer directly, the system first searches the user's document collection for relevant information.

The retrieved information is then provided to the LLM as context.

### Without RAG

User Question
      ↓
     LLM
      ↓
   Answer

### With RAG

User Question
      ↓
Semantic Search
      ↓
Relevant Document Chunks
      ↓
     LLM
      ↓
Context-Aware Answer

## System Architecture

<img width="2240" height="1786" alt="image" src="https://github.com/user-attachments/assets/5c381012-cf65-4b86-ab87-378e26981bf3" />

## 🔬 Technical Pipeline

### 1. Document Ingestion

PDF documents are loaded and converted into text using a PDF document loader.

### 2. Text Splitting

Large documents are divided into smaller chunks using a recursive text-splitting strategy.

Chunking allows the system to retrieve specific sections of a document instead of passing entire documents to the LLM.

### 3. Embedding Generation

Each document chunk is converted into a numerical vector representation using a Sentence Transformer embedding model.

### 4. Vector Storage

The generated embeddings are stored in ChromaDB.

The vector store uses cosine-based similarity to identify semantically similar content.

### 5. Query Processing

When a user submits a question, the question is converted into an embedding using the same embedding model.

### 6. Retrieval

The query embedding is compared against the stored document embeddings.

The system retrieves the most relevant chunks according to:

- Top-K
- Similarity threshold

### 7. Context Construction

The retrieved chunks are combined into a context that is passed to the language model.

### 8. Response Generation

The LLM generates a response based on the retrieved context.

### 9. User Interface

The entire pipeline is exposed through a Streamlit interface.

## 🎚️ Retrieval Controls

The application allows users to experiment with retrieval behaviour.

### Top-K

Controls the maximum number of document chunks retrieved for a query.

A higher value provides the model with more context, while a lower value can provide a more focused context.

### Similarity Threshold

Controls the minimum similarity score required for a retrieved chunk to be included.

A higher threshold makes retrieval more selective.

These parameters allow users to observe how retrieval configuration can influence the final generated response.

## 📚 Designed for Different Document Collections

The pipeline is not tied to a specific subject or dataset.

It can be used with different types of PDF documents, such as:

📖 Educational material  
📑 Research papers  
📋 Reports  
📚 Books and manuals  
🏢 Business documentation  
💻 Technical documentation  
📊 Project reports  

The underlying pipeline remains the same; only the document collection changes.
