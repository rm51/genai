from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from typing import Dict, List, Optional
from datasets import Dataset

# RAGAS imports
try:
    from ragas import SingleTurnSample
    from ragas.metrics import BleuScore, NonLLMContextPrecisionWithReference, ResponseRelevancy, Faithfulness, RougeScore
    from ragas import evaluate
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False

def evaluate_response_quality(question: str, answer: str, contexts: List[str], openai_key: str) -> Dict[str, float]:
    if not RAGAS_AVAILABLE:
        return {"error": "RAGAS not available"}
    
    voc_base_url = "https://openai.vocareum.com/v1"
    
    try:
        evaluator_llm = LangchainLLMWrapper(
            ChatOpenAI(
                model="gpt-3.5-turbo", 
                api_key=openai_key, 
                base_url=voc_base_url
            )
        )

        evaluator_embeddings = LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(
                model="text-embedding-3-small",
                api_key=openai_key,
                base_url=voc_base_url
            )
        )

        faithfulness = Faithfulness(llm=evaluator_llm)
        relevancy = ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings)

        dataset_dict = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts]
        }
        eval_dataset = Dataset.from_dict(dataset_dict)

        results = evaluate(
            dataset=eval_dataset,
            metrics=[faithfulness, relevancy]
        )

        return results.to_pandas().to_dict(orient="records")[0]

    except Exception as e:
        print(f"Error during RAGAS evaluation: {e}")
        return {"error": str(e)}