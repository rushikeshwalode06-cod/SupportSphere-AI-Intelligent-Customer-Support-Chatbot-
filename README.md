# 🤖 SupportSphere-AI-Intelligent-Customer-Support-Chatbot 

Customer Support AI is an intelligent RAG-based chatbot that uses Sentence Transformers, FAISS vector search, and FLAN-T5 Small to retrieve relevant information from a customer support knowledge base and generate accurate, context-aware answers to customer queries.

# 🤖 SupportSphere AI — Intelligent Customer Support Chatbot

## 📌 Project Overview

SupportSphere AI is an intelligent customer support chatbot built using Retrieval-Augmented Generation (RAG), Natural Language Processing (NLP), and Transformer-based models. It retrieves relevant information from a customer support knowledge base using semantic search and generates context-aware answers to customer queries.

The application combines Sentence Transformers, FAISS vector search, and FLAN-T5 Small to provide an interactive AI-powered customer support experience through a Streamlit web interface.

## 🎯 Objectives

- Build an AI-powered customer support chatbot.
- Retrieve relevant answers using semantic similarity.
- Generate context-aware responses using a text generation model.
- Reduce manual effort in answering common customer queries.
- Display relevant reference information to improve answer transparency.
- Develop an interactive chatbot using Streamlit.

## ✨ Key Features

- 🤖 **AI-Powered Chatbot:** Answers customer support questions.
- 🔍 **Semantic Search:** Finds relevant information based on query meaning.
- 🧠 **Text Embeddings:** Converts knowledge documents into numerical vectors.
- ⚡ **FAISS Vector Search:** Efficiently retrieves relevant documents.
- 💬 **Context-Aware Responses:** Generates answers using retrieved reference information.
- 📚 **Source References:** Displays related questions, reference answers, categories, and similarity scores.
- 🎛️ **Interactive Dashboard:** Includes a modern Streamlit interface.
- 🗂️ **Knowledge Base Integration:** Uses a customer support dataset.
- 🧹 **Chat Management:** Allows users to clear conversation history.
- ⚙️ **Configurable Retrieval:** Lets users adjust the number of retrieved sources.

## 🛠️ Technologies Used

1. 🐍 Python
2. 📝 Natural Language Processing (NLP)
3. 🔗 Retrieval-Augmented Generation (RAG)
4. 🧠 Sentence Transformers
5. 🤗 Hugging Face Transformers
6. 🗄️ FAISS
7. 🤖 FLAN-T5 Small
8. 🔢 NumPy
9. 🐼 Pandas
10. ⚡ PyTorch
11. 🖥️ Streamlit
12. ☁️ Google Colab
13. 📚 Hugging Face Datasets

## 🧠 Model Architecture

- 🧠 **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2`
- 🤖 **Text Generation Model:** `google/flan-t5-small`
- 🗄️ **Vector Search:** FAISS `IndexFlatIP`
- 📐 **Retrieval Method:** Cosine similarity using normalized embeddings and inner-product search.
- 🔗 **Architecture Type:** Retrieval-Augmented Generation (RAG)
- 🖥️ **Application Framework:** Streamlit

## 📂 Dataset

The project uses the **Bitext Customer Support LLM Chatbot Training Dataset**, available through Hugging Face.

**Dataset Repository:** `bitext/Bitext-customer-support-llm-chatbot-training-dataset`

The dataset contains customer support questions and corresponding reference responses, along with category and intent information.

### 📊 Dataset Processing

1. 📥 **Load Dataset:** Load the customer support dataset.
2. ✅ **Column Verification:** Verify the required columns.
3. 🧹 **Data Cleaning:** Remove missing and empty questions or responses.
4. 🔄 **Duplicate Removal:** Remove duplicate question-answer pairs.
5. 📄 **Knowledge Document Creation:** Create knowledge documents using category, intent, customer question, and reference answer.
6. 🧠 **Embedding Generation:** Generate embeddings for the knowledge documents.
7. 🗄️ **Vector Indexing:** Store the embeddings in a FAISS vector index.

## ⚙️ Project Workflow

1. 📥 **Data Collection:** Load the customer support dataset.
2. 🧹 **Data Preprocessing:** Clean missing values, empty records, and duplicate entries.
3. 📄 **Document Creation:** Combine customer questions, responses, categories, and intents into knowledge documents.
4. 🧠 **Embedding Generation:** Convert knowledge documents into vector representations using Sentence Transformers.
5. 🗄️ **Vector Indexing:** Store the embeddings in a FAISS index.
6. 💬 **Query Processing:** Accept a customer question through the chatbot interface.
7. 🔍 **Semantic Retrieval:** Retrieve the most relevant documents using vector similarity.
8. 📝 **Context Preparation:** Combine the retrieved documents into reference information.
9. 🤖 **Answer Generation:** Provide the question and retrieved context to FLAN-T5 Small.
10. 📊 **Response Display:** Display the generated answer and relevant source references in Streamlit.

## 🔎 How the RAG System Works

The chatbot follows a Retrieval-Augmented Generation pipeline:

**Customer Query → Query Embedding → FAISS Similarity Search → Relevant Documents → Context Preparation → FLAN-T5 Generation → Final Answer**

Instead of relying only on the generation model's internal knowledge, the chatbot uses retrieved customer support information as context for answering questions.

The prompt instructs the model to use the supplied references and avoid inventing company policies or unsupported facts.

## 📊 Model Components

| Component | Model / Technique | Purpose |
|---|---|---|
| 🧠 Embedding Model | all-MiniLM-L6-v2 | Converts text into embeddings |
| 🗄️ Vector Database | FAISS IndexFlatIP | Retrieves similar documents |
| 📐 Similarity Method | Cosine Similarity | Measures semantic relevance |
| 🤖 Generation Model | FLAN-T5 Small | Generates customer support answers |
| ⚙️ Deep Learning Framework | PyTorch | Runs the generation model |
| 🖥️ Application Framework | Streamlit | Provides the chatbot interface |

## 💻 Example

**Customer Question:**

How can I cancel my order?

## ⚙️ System Process

1. 🧮 Converts the customer question into an embedding.
2. 🔍 Searches the FAISS index for relevant knowledge documents.
3. 📚 Retrieves the top-matching customer support references.
4. 🧠 Supplies the retrieved context to FLAN-T5 Small.
5. 🤖 Generates an answer based on the available reference information.

**Output:**

The chatbot displays the generated answer and, when enabled, the retrieved source details.

*Note: The actual response depends on the retrieved dataset records.*

## 🖥️ Application Interface

The Streamlit application includes:

1. 📊 A SupportSphere AI dashboard.
2. 📈 Knowledge record and indexed vector statistics.
3. 💬 A customer question input field.
4. 🚀 An interactive Ask SupportSphere button.
5. 🗂️ A persistent question-and-answer history.
6. 📚 An expandable section containing retrieved source references.
7. ⚙️ Sidebar settings for retrieval count and source visibility.
8. 🧹 A clear-chat option.
## ⚠️ Limitations

1. 📚 Answer quality depends on the relevance and completeness of the knowledge base.
2. 🤖 FLAN-T5 Small may generate incomplete or imperfect responses.
3. 🔍 Semantic retrieval may return related but not fully relevant documents.
4. 🌐 The project uses a public customer support dataset and does not automatically know a particular company's current policies.
5. 🧪 The application is a prototype and should be tested before being used in a production customer support environment.

## 🔮 Future Scope

1. 🏢 Integrate company-specific knowledge bases and FAQs.
2. 📄 Support PDF, DOCX, and website content ingestion.
3. 🌐 Add multilingual customer support.
4. 🔍 Improve retrieval using hybrid search and reranking.
5. 🤖 Experiment with larger language models.
6. 🧠 Add conversation memory and personalized support.
7. 🔐 Introduce user authentication and chat history storage.
8. 💬 Add response feedback and quality evaluation.
9. ☁️ Deploy the application on a scalable cloud platform.
10. 📊 Evaluate retrieval quality and answer faithfulness using suitable metrics.

## 🏁 Conclusion

SupportSphere AI demonstrates how Retrieval-Augmented Generation can combine semantic search and text generation to build an intelligent customer support chatbot. By integrating Sentence Transformers, FAISS, FLAN-T5 Small, and Streamlit, the project retrieves relevant knowledge and generates context-aware responses through an interactive interface. It provides a foundation for developing more advanced AI-driven customer support systems.
