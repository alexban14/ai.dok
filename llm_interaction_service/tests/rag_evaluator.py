"""
Evaluation script for RAG system.
Measures Precision@K, Recall@K, MRR, and hallucination rate.
"""

import json
import logging
from typing import List, Dict, Tuple
from pathlib import Path
import asyncio
from dataclasses import dataclass
import re

logger = logging.getLogger(__name__)

@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics"""
    precision_at_5: float
    recall_at_5: float
    mrr: float  # Mean Reciprocal Rank
    hallucination_rate: float
    section_accuracy: float
    avg_response_time: float
    total_queries: int

class RAGEvaluator:
    """
    Evaluator for RAG system performance.
    Compares retrieved documents and generated answers against ground truth.
    """
    
    def __init__(self, test_set_path: str):
        """
        Initialize evaluator with test dataset.
        
        Args:
            test_set_path: Path to test set JSON file
        """
        self.test_set_path = test_set_path
        self.test_data = self._load_test_set()
        logger.info(f"Loaded {len(self.test_data['queries'])} test queries")
    
    def _load_test_set(self) -> Dict:
        """Load test set from JSON file"""
        with open(self.test_set_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def evaluate_retrieval(
        self,
        retrieved_docs: List[Dict],
        expected_section: str,
        expected_drug: str,
        k: int = 5
    ) -> Dict[str, float]:
        """
        Evaluate retrieval quality for a single query.
        
        Args:
            retrieved_docs: List of retrieved documents with metadata
            expected_section: Expected RCP section number
            expected_drug: Expected drug name
            k: Number of top documents to evaluate
            
        Returns:
            Dictionary with precision, recall, reciprocal_rank, section_match
        """
        # Check if expected section appears in top-k
        relevant_docs = []
        first_relevant_rank = None
        
        for rank, doc in enumerate(retrieved_docs[:k], start=1):
            metadata = doc.get('metadata', {})
            section_number = metadata.get('section_number', '')
            source = metadata.get('source', '')
            
            # Check if document is relevant
            is_relevant = (
                section_number.startswith(expected_section) or
                expected_section in section_number or
                expected_drug.lower() in source.lower()
            )
            
            if is_relevant:
                relevant_docs.append(doc)
                if first_relevant_rank is None:
                    first_relevant_rank = rank
        
        # Calculate metrics
        precision = len(relevant_docs) / k if k > 0 else 0
        # Recall: assuming 1 relevant document exists
        recall = 1.0 if len(relevant_docs) > 0 else 0.0
        reciprocal_rank = 1.0 / first_relevant_rank if first_relevant_rank else 0.0
        section_match = any(
            doc.get('metadata', {}).get('section_number', '').startswith(expected_section)
            for doc in retrieved_docs[:k]
        )
        
        return {
            'precision': precision,
            'recall': recall,
            'reciprocal_rank': reciprocal_rank,
            'section_match': float(section_match),
            'relevant_count': len(relevant_docs)
        }
    
    def detect_hallucination(
        self,
        generated_answer: str,
        retrieved_context: str,
        ground_truth: str
    ) -> bool:
        """
        Detect if generated answer contains hallucinated information.
        
        Args:
            generated_answer: LLM-generated answer
            retrieved_context: Retrieved document context
            ground_truth: Expected answer
            
        Returns:
            True if hallucination detected, False otherwise
        """
        # Simple heuristic: check if answer contains key terms not in context
        # Extract meaningful terms from answer (longer than 3 chars, excluding common words)
        common_words = {
            'este', 'sunt', 'pentru', 'with', 'the', 'and', 'sau', 'la', 'de',
            'în', 'cu', 'pe', 'un', 'una', 'din', 'care', 'poate', 'trebuie'
        }
        
        # Extract key terms from answer
        answer_terms = set(
            term.lower() 
            for term in re.findall(r'\b\w{4,}\b', generated_answer)
            if term.lower() not in common_words
        )
        
        # Extract terms from context
        context_terms = set(
            term.lower()
            for term in re.findall(r'\b\w{4,}\b', retrieved_context)
            if term.lower() not in common_words
        )
        
        # Check for "Nu există informații" or similar (non-hallucination indicator)
        if any(phrase in generated_answer.lower() for phrase in [
            'nu există', 'nu am găsit', 'informații insuficiente',
            'nu pot găsi', 'nu este disponibil'
        ]):
            return False  # Model correctly stated lack of info
        
        # Calculate how many answer terms are missing from context
        missing_terms = answer_terms - context_terms
        missing_ratio = len(missing_terms) / len(answer_terms) if answer_terms else 0
        
        # If >50% of key terms are not in context, likely hallucination
        is_hallucination = missing_ratio > 0.5
        
        if is_hallucination:
            logger.debug(f"Hallucination detected. Missing terms: {missing_terms}")
        
        return is_hallucination
    
    def evaluate_answer_quality(
        self,
        generated_answer: str,
        ground_truth: str,
        retrieved_context: str
    ) -> Dict[str, float]:
        """
        Evaluate quality of generated answer.
        
        Args:
            generated_answer: LLM-generated answer
            ground_truth: Expected answer
            retrieved_context: Retrieved context
            
        Returns:
            Dictionary with quality metrics
        """
        # Check for hallucination
        has_hallucination = self.detect_hallucination(
            generated_answer,
            retrieved_context,
            ground_truth
        )
        
        # Check for key terms from ground truth in generated answer
        ground_truth_terms = set(
            term.lower()
            for term in re.findall(r'\b\w{4,}\b', ground_truth)
        )
        answer_terms = set(
            term.lower()
            for term in re.findall(r'\b\w{4,}\b', generated_answer)
        )
        
        # Calculate overlap
        overlap = len(ground_truth_terms & answer_terms)
        coverage = overlap / len(ground_truth_terms) if ground_truth_terms else 0
        
        return {
            'has_hallucination': float(has_hallucination),
            'term_coverage': coverage,
            'answer_length': len(generated_answer)
        }
    
    async def evaluate_system(
        self,
        rag_service,
        strategy: str = "hybrid",
        k: int = 5
    ) -> EvaluationMetrics:
        """
        Evaluate full RAG system on test set.
        
        Args:
            rag_service: RAG service instance
            strategy: Retrieval strategy to test
            k: Number of documents to retrieve
            
        Returns:
            EvaluationMetrics object
        """
        total_precision = 0.0
        total_recall = 0.0
        total_mrr = 0.0
        total_hallucinations = 0
        total_section_matches = 0
        total_time = 0.0
        
        queries = self.test_data['queries']
        
        for i, query_data in enumerate(queries, 1):
            query = query_data['query']
            expected_section = query_data.get('expected_section', '')
            expected_drug = query_data.get('expected_drug', '')
            ground_truth = query_data.get('ground_truth_answer', '')
            
            logger.info(f"Evaluating query {i}/{len(queries)}: {query}")
            
            try:
                # Execute query
                import time
                start_time = time.time()
                
                # Call RAG service (mock for now - replace with actual call)
                # result = await rag_service.query(
                #     model="llama-3.3-70b-versatile",
                #     prompt=query,
                #     ai_service="groq_cloud",
                #     collection_name="rcp_documents_v2",
                #     retrieval_strategy=strategy
                # )
                
                # For now, simulate result structure
                result = {
                    'retrieved_docs': [],
                    'answer': '',
                    'context': ''
                }
                
                elapsed_time = time.time() - start_time
                total_time += elapsed_time
                
                # Evaluate retrieval
                retrieval_metrics = self.evaluate_retrieval(
                    result.get('retrieved_docs', []),
                    expected_section,
                    expected_drug,
                    k
                )
                
                total_precision += retrieval_metrics['precision']
                total_recall += retrieval_metrics['recall']
                total_mrr += retrieval_metrics['reciprocal_rank']
                total_section_matches += retrieval_metrics['section_match']
                
                # Evaluate answer quality
                if ground_truth and result.get('answer'):
                    answer_metrics = self.evaluate_answer_quality(
                        result['answer'],
                        ground_truth,
                        result.get('context', '')
                    )
                    total_hallucinations += answer_metrics['has_hallucination']
                
            except Exception as e:
                logger.error(f"Error evaluating query {i}: {e}")
                continue
        
        # Calculate average metrics
        n = len(queries)
        metrics = EvaluationMetrics(
            precision_at_5=total_precision / n,
            recall_at_5=total_recall / n,
            mrr=total_mrr / n,
            hallucination_rate=total_hallucinations / n,
            section_accuracy=total_section_matches / n,
            avg_response_time=total_time / n,
            total_queries=n
        )
        
        return metrics
    
    def compare_strategies(
        self,
        rag_service,
        strategies: List[str] = ["vector_only", "hybrid"],
        k: int = 5
    ) -> Dict[str, EvaluationMetrics]:
        """
        Compare different retrieval strategies.
        
        Args:
            rag_service: RAG service instance
            strategies: List of strategies to compare
            k: Number of documents
            
        Returns:
            Dictionary mapping strategy name to metrics
        """
        results = {}
        
        for strategy in strategies:
            logger.info(f"\n{'='*60}")
            logger.info(f"Evaluating strategy: {strategy}")
            logger.info(f"{'='*60}\n")
            
            metrics = asyncio.run(
                self.evaluate_system(rag_service, strategy, k)
            )
            results[strategy] = metrics
            
            # Print results
            self._print_metrics(strategy, metrics)
        
        return results
    
    def _print_metrics(self, strategy: str, metrics: EvaluationMetrics):
        """Print evaluation metrics in formatted table"""
        print(f"\n{'='*60}")
        print(f"Results for {strategy.upper()}")
        print(f"{'='*60}")
        print(f"Total Queries:        {metrics.total_queries}")
        print(f"Precision@5:          {metrics.precision_at_5:.3f}")
        print(f"Recall@5:             {metrics.recall_at_5:.3f}")
        print(f"MRR:                  {metrics.mrr:.3f}")
        print(f"Hallucination Rate:   {metrics.hallucination_rate:.3f}")
        print(f"Section Accuracy:     {metrics.section_accuracy:.3f}")
        print(f"Avg Response Time:    {metrics.avg_response_time:.3f}s")
        print(f"{'='*60}\n")
    
    def save_results(
        self,
        results: Dict[str, EvaluationMetrics],
        output_path: str
    ):
        """
        Save evaluation results to JSON file.
        
        Args:
            results: Dictionary of strategy -> metrics
            output_path: Path to save results
        """
        output_data = {
            'evaluation_date': str(Path(output_path).stat().st_mtime),
            'strategies': {}
        }
        
        for strategy, metrics in results.items():
            output_data['strategies'][strategy] = {
                'precision_at_5': metrics.precision_at_5,
                'recall_at_5': metrics.recall_at_5,
                'mrr': metrics.mrr,
                'hallucination_rate': metrics.hallucination_rate,
                'section_accuracy': metrics.section_accuracy,
                'avg_response_time': metrics.avg_response_time,
                'total_queries': metrics.total_queries
            }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {output_path}")


# Usage example
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize evaluator
    evaluator = RAGEvaluator(
        test_set_path="tests/data/rag_test_set.json"
    )
    
    # Mock RAG service for testing
    class MockRAGService:
        pass
    
    rag_service = MockRAGService()
    
    # Run evaluation
    results = evaluator.compare_strategies(
        rag_service,
        strategies=["vector_only", "hybrid"],
        k=5
    )
    
    # Save results
    evaluator.save_results(
        results,
        output_path="tests/results/evaluation_results.json"
    )
