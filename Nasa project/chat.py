#!/usr/bin/env python3
"""
NASA RAG Chat with RAGAS Evaluation Integration (With Batch Testing)
"""

import streamlit as st
import os
import json
import pandas as pd

import ragas_evaluator
import rag_client
import llm_client

from pathlib import Path
from typing import Dict, List, Optional

# RAGAS imports
try:
    from ragas import SingleTurnSample
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="NASA RAG Chat with Evaluation",
    page_icon="🚀",
    layout="wide"
)

def discover_chroma_backends() -> Dict[str, Dict[str, str]]:
    """Discover available ChromaDB backends in the project directory"""
    return rag_client.discover_chroma_backends()

def initialize_rag_system(chroma_dir: str, collection_name: str):
    """Initialize the RAG system with specified backend"""
    try:
       return rag_client.initialize_rag_system(chroma_dir, collection_name)
    except Exception as e:
        return None, False, str(e)

def retrieve_documents(collection, query: str, n_results: int = 3, 
                      mission_filter: Optional[str] = None) -> Optional[Dict]:
    """Retrieve relevant documents from ChromaDB"""
    try:
        return rag_client.retrieve_documents(collection, query, n_results, mission_filter)
    except Exception as e:
        st.error(f"Error retrieving documents: {e}")
        return None

def format_context(documents: List[str], metadatas: List[Dict]) -> str:
    """Format retrieved documents into context"""
    return rag_client.format_context(documents, metadatas)

def generate_response(openai_key, user_message: str, context: str, 
                     conversation_history: List[Dict], model: str = "gpt-3.5-turbo") -> str:
    """Generate response using OpenAI with context"""
    try:
        return llm_client.generate_response(openai_key, user_message, context, conversation_history, model)
    except Exception as e:
        return f"Error generating response: {e}"

def evaluate_response_quality(question: str, answer: str, contexts: List[str], openai_key: str) -> Dict[str, float]:
    """Evaluate response quality using RAGAS metrics"""
    try:
        return ragas_evaluator.evaluate_response_quality(question, answer, contexts, openai_key)
    except Exception as e:
        return {"error": f"Evaluation failed: {str(e)}"}

def display_evaluation_metrics(scores: Dict[str, float]):
    """Display evaluation metrics in the sidebar"""
    if "error" in scores:
        st.sidebar.error(f"Evaluation Error: {scores['error']}")
        return
    
    st.sidebar.subheader("📊 Response Quality")
    for metric_name, score in scores.items():
        if isinstance(score, (int, float)):
            safe_score = max(0.0, min(1.0, float(score)))
            st.sidebar.metric(
                label=metric_name.replace('_', ' ').title(),
                value=f"{safe_score:.3f}"
            )
            st.sidebar.progress(safe_score)

def main():
    st.title("🚀 NASA Space Mission Chat with Evaluation")
    st.markdown("Chat with AI about NASA space missions with real-time quality evaluation")
    
    if not RAGAS_AVAILABLE:
        st.warning("RAGAS framework package not found natively. Install via terminal: pip install ragas")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_backend" not in st.session_state:
        st.session_state.current_backend = None
    if "last_evaluation" not in st.session_state:
        st.session_state.last_evaluation = None
    if "last_contexts" not in st.session_state:
        st.session_state.last_contexts = []
    
    # -------------------------------------------------------------
    # Sidebar: Core Configuration
    # -------------------------------------------------------------
    with st.sidebar:
        st.header("🔧 Configuration")
        
        openai_key = st.text_input("OpenAI API Key", type="password")
        if not openai_key:
            st.info("Please enter your OpenAI API key to continue.")
            st.stop()

        with st.spinner("Discovering ChromaDB backends..."):
            available_backends = discover_chroma_backends()
        
        if not available_backends:
            st.error("No ChromaDB backends found!")
            st.stop()
        
        st.subheader("📊 ChromaDB Backend")
        backend_options = {k: v["display_name"] for k, v in available_backends.items()}
        
        selected_backend_key = st.selectbox(
            "Select Dataset", 
            options=list(backend_options.keys()), 
            format_func=lambda x: backend_options[x]
        )
        
        backend_info = available_backends[selected_backend_key]
        
        model_choice = st.selectbox("Model", ["gpt-3.5-turbo", "gpt-4"], index=0)
        n_docs = st.slider("Documents to retrieve", 1, 10, 3)
        enable_evaluation = st.checkbox("Enable RAGAS Evaluation", value=True)

    # -------------------------------------------------------------
    # RAG System Initialization (Must happen BEFORE batch execution block)
    # -------------------------------------------------------------
    db_path = backend_info.get("path") or backend_info.get("directory")
    coll_name = backend_info.get("collection") or backend_info.get("collection_name")
    collection, success, message = initialize_rag_system(db_path, coll_name)
    if not success:
        st.sidebar.error(f"Failed to initialize RAG backend: {message}")
        st.stop()

    # -------------------------------------------------------------
    # Sidebar: Batch Evaluation Dataset Execution Block
    # -------------------------------------------------------------
    with st.sidebar:
        st.markdown("---")
        st.subheader("🧪 Batch Testing & Dataset Evaluation")
        test_dataset_path = "test_questions.json"
        
        if st.button("Run Batch Evaluation"):
            if not RAGAS_AVAILABLE:
                st.error("Cannot run batch jobs without RAGAS. Run 'pip install ragas' in terminal first.")
            elif not os.path.exists(test_dataset_path):
                st.error(f"Error: `{test_dataset_path}` data file is missing. Ensure it exists in your directory.")
            else:
                with st.spinner("Loading test dataset and running batch evaluation..."):
                    try:
                        with open(test_dataset_path, "r") as f:
                            test_cases = json.load(f)
                    except json.JSONDecodeError:
                        st.error("Error: `test_questions.json` is empty or corrupted.")
                        test_cases = []
                    except Exception as e:
                        st.error(f"Failed to load dataset file: {str(e)}")
                        test_cases = []

                    if test_cases:
                        batch_results = []
                        progress_bar = st.progress(0)
                        total_cases = len(test_cases)
                        
                        for idx, case in enumerate(test_cases):
                            q = case.get("question", "").strip()
                            if not q:
                                continue  # Cleanly skip empty data lines
                            
                            # Retrieve chunks using the valid collection variable
                            docs_result = retrieve_documents(collection, q, n_docs)
                            context = ""
                            contexts_list = []
                            if docs_result and docs_result.get("documents"):
                                context = format_context(docs_result["documents"][0], docs_result["metadatas"][0])
                                contexts_list = docs_result["documents"][0]
                            
                            # Generate response
                            response = generate_response(openai_key, q, context, [], model_choice)
                            
                            # Run evaluation metrics
                            scores = evaluate_response_quality(q, str(response), contexts_list, openai_key)
                            
                            row_data = {
                                "Question": q,
                                "Category": case.get("category", "Unspecified"),
                                "Answer": str(response),
                                "Faithfulness": scores.get("faithfulness", 0.0) if "error" not in scores else 0.0,
                                "Answer Relevancy": scores.get("answer_relevancy", 0.0) if "error" not in scores else 0.0
                            }
                            batch_results.append(row_data)
                            progress_bar.progress((idx + 1) / total_cases)
                        
                        if batch_results:
                            st.session_state.batch_results_df = pd.DataFrame(batch_results)
                            st.success("Batch metrics compiled successfully!")
                        else:
                            st.error("No valid questions parsed from the tracking dataset.")

    # -------------------------------------------------------------
    # Main Panel Dashboard UI
    # -------------------------------------------------------------
    if "batch_results_df" in st.session_state:
        st.markdown("## 📊 Batch Evaluation Summary")
        df = st.session_state.batch_results_df
        
        # Calculate Aggregates
        mean_faith = df["Faithfulness"].mean()
        mean_relevancy = df["Answer Relevancy"].mean()
        
        m_col1, m_col2 = st.columns(2)
        m_col1.metric("Dataset Mean Faithfulness", f"{mean_faith:.3f}")
        m_col2.metric("Dataset Mean Answer Relevancy", f"{mean_relevancy:.3f}")
        
        # Interactive Summary Table
        st.dataframe(df[["Question", "Category", "Faithfulness", "Answer Relevancy"]], use_container_width=True)
        
        # Comprehensive individual log break-outs
        with st.expander("See comprehensive responses and scores per test query"):
            for _, row in df.iterrows():
                st.markdown(f"**Q: {row['Question']}** *(Category: {row['Category']})*")
                st.markdown(f"**A:** {row['Answer']}")
                st.markdown(f"📈 *Faithfulness:* `{row['Faithfulness']:.3f}` | *Answer Relevancy:* `{row['Answer Relevancy']:.3f}`")
                st.markdown("---")
                
        if st.button("Clear Batch Dashboard View"):
            del st.session_state.batch_results_df
            st.rerun()

    st.markdown("---")

    # Display interactive chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat Input Interface
    if prompt := st.chat_input("Ask about NASA space missions..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Searching and generating..."):
                docs_result = retrieve_documents(collection, prompt, n_docs)
                
                context = ""
                contexts_list = []
                if docs_result and docs_result.get("documents"):
                    context = format_context(docs_result["documents"][0], docs_result["metadatas"][0])
                    contexts_list = docs_result["documents"][0]
                    st.session_state.last_contexts = contexts_list
                
                response = generate_response(openai_key, prompt, context, st.session_state.messages[:-1], model_choice)
                response_text = str(response)
                st.markdown(response_text)
                
                if enable_evaluation and RAGAS_AVAILABLE:
                    with st.spinner("Calculating response quality..."):
                        evaluation_scores = evaluate_response_quality(prompt, response_text, contexts_list, openai_key)
                        st.session_state.last_evaluation = evaluation_scores
        
        st.session_state.messages.append({"role": "assistant", "content": response_text})

    # Show live metrics in the sidebar frame for individual chats
    if st.session_state.last_evaluation:
        display_evaluation_metrics(st.session_state.last_evaluation)

if __name__ == "__main__":
    main()