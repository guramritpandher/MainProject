import os
import fitz  # PyMuPDF
import faiss
import numpy as np
import torch
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from nltk.tokenize import sent_tokenize
import nltk

# Ensure NLTK data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class EnhancedPDFChatbot:
    """
    Enhanced PDF Chatbot with improved accuracy through better models,
    hybrid retrieval, and advanced prompt engineering.
    """
    def __init__(self, pdf_path, 
                 embedding_model_name="sentence-transformers/all-mpnet-base-v2",  # Better semantic understanding
                 llm_model_name="google/flan-t5-large",  # More powerful LLM
                 chunk_size=500, 
                 chunk_overlap=150, 
                 top_k=8,
                 use_hybrid_search=True):
        """
        Initialize the enhanced chatbot with better defaults
        
        Args:
            pdf_path: Path to the PDF file
            embedding_model_name: Model for text embeddings
            llm_model_name: Model for answer generation
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            top_k: Number of chunks to retrieve
            use_hybrid_search: Whether to use hybrid search (dense + sparse)
        """
        self.pdf_path = pdf_path
        self.embedding_model_name = embedding_model_name
        self.llm_model_name = llm_model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.use_hybrid_search = use_hybrid_search
        
        # Load embedding model
        print(f"Loading embedding model: {embedding_model_name}")
        self.embedding_model = HuggingFaceEmbeddings(model_name=self.embedding_model_name)
        
        # Load LLM
        print(f"Loading LLM: {llm_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)
        
        # Check if we're using a causal LM or seq2seq model
        if "t5" in self.llm_model_name.lower():
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.llm_model_name, 
                device_map="auto",
                torch_dtype=torch.float16  # Use half precision for memory efficiency
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.llm_model_name,
                device_map="auto",
                torch_dtype=torch.float16
            )
        
        # Process the PDF
        print("Extracting text from PDF...")
        self.raw_text = self.extract_text_from_pdf()
        
        # Create chunks with metadata
        print("Creating text chunks...")
        self.chunks, self.chunk_metadata = self.create_text_chunks_with_metadata()
        
        # Create FAISS index
        print("Creating FAISS index...")
        self.index = self.create_faiss_index()
        
        # Create TF-IDF vectorizer for hybrid search
        if self.use_hybrid_search:
            print("Creating TF-IDF vectorizer...")
            self.tfidf_vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform([chunk for chunk in self.chunks])
        
        print("Chatbot initialization complete!")

    def extract_text_from_pdf(self):
        """Extract text from PDF with improved structure preservation"""
        text = ""
        doc = fitz.open(self.pdf_path)
        
        for page_num, page in enumerate(doc):
            # Extract text with blocks to better preserve structure
            blocks = page.get_text("blocks")
            page_text = ""
            
            for block in blocks:
                block_text = block[4]
                # Add section markers for headings (heuristic: short lines with few words)
                words = block_text.split()
                if len(words) <= 8 and len(block_text) <= 100 and block_text.strip():
                    page_text += f"\n## {block_text.strip()} ##\n"
                else:
                    page_text += block_text + "\n"
            
            # Add page metadata
            text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
        
        return text.strip()

    def create_text_chunks_with_metadata(self):
        """Split text into chunks with metadata about their source location"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap,
            separators=["\n## ", "\n--- Page", "\n\n", "\n", ". ", " ", ""]
        )
        
        chunks = text_splitter.split_text(self.raw_text)
        
        # Extract metadata for each chunk
        chunk_metadata = []
        for i, chunk in enumerate(chunks):
            # Extract page number if present
            page_match = re.search(r"--- Page (\d+) ---", chunk)
            page_num = int(page_match.group(1)) if page_match else None
            
            # Extract section title if present
            section_match = re.search(r"## (.*?) ##", chunk)
            section = section_match.group(1) if section_match else None
            
            # Clean the chunk for actual use (remove metadata markers)
            chunks[i] = re.sub(r"--- Page \d+ ---", "", chunk)
            chunks[i] = re.sub(r"## (.*?) ##", r"\1", chunks[i])
            
            # Store metadata
            chunk_metadata.append({
                "page": page_num,
                "section": section,
                "index": i
            })
        
        return chunks, chunk_metadata

    def create_faiss_index(self):
        """Create FAISS index for fast similarity search"""
        # Generate embeddings for all chunks
        embeddings = np.array([
            self.embedding_model.embed_query(text) for text in self.chunks
        ], dtype=np.float32)
        
        # Get embedding dimension
        d = embeddings.shape[1]
        
        # Create index - using IndexFlatIP for inner product (cosine similarity)
        index = faiss.IndexFlatIP(d)
        
        # Add vectors to index
        index.add(embeddings)
        
        return index

    def hybrid_search(self, query, alpha=0.7):
        """
        Perform hybrid search combining dense and sparse retrieval
        
        Args:
            query: The user query
            alpha: Weight for dense retrieval (1-alpha for sparse)
            
        Returns:
            List of chunk indices sorted by relevance
        """
        # Dense retrieval (vector similarity)
        query_embedding = np.array([self.embedding_model.embed_query(query)], dtype=np.float32)
        dense_distances, dense_indices = self.index.search(query_embedding, self.top_k * 2)
        
        # Convert to dictionary for easier manipulation
        dense_scores = {idx: score for idx, score in zip(dense_indices[0], dense_distances[0])}
        
        # Sparse retrieval (TF-IDF)
        query_tfidf = self.tfidf_vectorizer.transform([query])
        sparse_scores = np.zeros(len(self.chunks))
        
        # Calculate dot product between query and all documents
        for i in range(len(self.chunks)):
            sparse_scores[i] = query_tfidf.dot(self.tfidf_matrix[i].T)[0, 0]
        
        # Get top sparse results
        sparse_top_indices = np.argsort(-sparse_scores)[:self.top_k * 2]
        sparse_top_scores = {idx: sparse_scores[idx] for idx in sparse_top_indices}
        
        # Normalize scores
        if dense_scores:
            max_dense = max(dense_scores.values())
            min_dense = min(dense_scores.values())
            dense_range = max_dense - min_dense
            if dense_range > 0:
                dense_scores = {k: (v - min_dense) / dense_range for k, v in dense_scores.items()}
        
        if sparse_top_scores:
            max_sparse = max(sparse_top_scores.values())
            min_sparse = min(sparse_top_scores.values())
            sparse_range = max_sparse - min_sparse
            if sparse_range > 0:
                sparse_top_scores = {k: (v - min_sparse) / sparse_range for k, v in sparse_top_scores.items()}
        
        # Combine scores
        combined_scores ={}
        all_indices = set(list(dense_scores.keys()) + list(sparse_top_scores.keys()))
        
        for idx in all_indices:
            dense_score = dense_scores.get(idx, 0)
            sparse_score = sparse_top_scores.get(idx, 0)
            combined_scores[idx] = alpha * dense_score + (1 - alpha) * sparse_score
        
        # Sort by combined score
        sorted_indices = sorted(combined_scores.keys(), key=lambda idx: combined_scores[idx], reverse=True)
        
        return sorted_indices[:self.top_k]

    def retrieve_relevant_chunks(self, query):
        """Retrieve most relevant chunks for the query"""
        if self.use_hybrid_search:
            # Use hybrid search (dense + sparse)
            top_indices = self.hybrid_search(query)
        else:
            # Use only dense retrieval
            query_embedding = np.array([self.embedding_model.embed_query(query)], dtype=np.float32)
            _, indices = self.index.search(query_embedding, self.top_k)
            top_indices = indices[0]
        
        # Get the actual chunks
        retrieved_chunks = []
        for idx in top_indices:
            if idx < len(self.chunks):
                chunk = self.chunks[idx]
                metadata = self.chunk_metadata[idx]
                
                # Format chunk with metadata
                formatted_chunk = f"[Page {metadata['page']}]"
                if metadata['section']:
                    formatted_chunk += f" [Section: {metadata['section']}]"
                formatted_chunk += f"\n{chunk}\n"
                
                retrieved_chunks.append(formatted_chunk)
        
        # Join chunks and clean
        context = "\n".join(retrieved_chunks)
        context = self._clean_context(context)
        
        return context

    def _clean_context(self, context):
        """Clean and deduplicate context"""
        # Split into paragraphs
        paragraphs = context.split('\n')
        
        # Remove duplicates while preserving order
        seen = set()
        cleaned_paragraphs = []
        
        for para in paragraphs:
            para = para.strip()
            # Skip empty lines and exact duplicates
            if not para or para in seen:
                continue
                
            # Add to seen set and cleaned list
            seen.add(para)
            cleaned_paragraphs.append(para)
        
        # Join back into text
        cleaned_context = '\n'.join(cleaned_paragraphs)
        
        return cleaned_context

    def generate_dynamic_prompt(self, query, context):
        """
        Generate a dynamic prompt based on query type
        """
        # Analyze query type
        query_lower = query.lower()
        
        # Check for different question types
        if any(word in query_lower for word in ["what is", "define", "explain", "describe"]):
            prompt_type = "definition"
        elif any(word in query_lower for word in ["how to", "how do", "steps", "process"]):
            prompt_type = "procedure"
        elif any(word in query_lower for word in ["why", "reason", "cause"]):
            prompt_type = "explanation"
        elif any(word in query_lower for word in ["compare", "difference", "versus", "vs"]):
            prompt_type = "comparison"
        elif any(word in query_lower for word in ["list", "examples", "types", "characteristics"]):
            prompt_type = "listing"
        else:
            prompt_type = "general"
        
        # Base prompt template
        base_template = f"""
        You are an AI assistant specialized in answering questions about documents.
        
        Below is an excerpt from a document, with page numbers and section information where available.
        Use ONLY the information from this excerpt to answer the question.
        If the information to answer the question is not in the excerpt, say "I don't have enough information to answer this question based on the document."
        
        Document excerpt:
        {context}
        
        Question: {query}
        """
        
        # Add specialized instructions based on query type
        if prompt_type == "definition":
            base_template += "\nProvide a clear and concise definition or explanation based on the document."
        elif prompt_type == "procedure":
            base_template += "\nExplain the process or steps in a clear, sequential manner as described in the document."
        elif prompt_type == "explanation":
            base_template += "\nExplain the reasons or causes as mentioned in the document."
        elif prompt_type == "comparison":
            base_template += "\nCompare and contrast the items mentioned in the question, highlighting key differences and similarities from the document."
        elif prompt_type == "listing":
            base_template += "\nProvide a comprehensive list of all relevant items mentioned in the document."
        
        # Add general instruction for all types
        base_template += "\n\nAnswer:"
        
        return base_template

    def generate_answer(self, query):
        """Generate an answer to the user's query"""
        # Retrieve relevant context
        context = self.retrieve_relevant_chunks(query)
        
        # If no relevant context found
        if not context.strip():
            return "I couldn't find relevant information in the document to answer your question."
        
        # Generate dynamic prompt
        prompt = self.generate_dynamic_prompt(query, context)
        
        # Prepare inputs for the model
        if "t5" in self.llm_model_name.lower():
            # For T5 models
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True).to(self.model.device)
            
            with torch.no_grad():
                output_ids = self.model.generate(
                    inputs.input_ids,
                    max_length=512,
                    min_length=50,
                    do_sample=True,
                    top_k=50,
                    top_p=0.85,
                    temperature=0.5,
                    num_return_sequences=1,
                    no_repeat_ngram_size=3
                )
            
            answer = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        else:
            # For causal models (GPT-style)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                output = self.model.generate(
                    inputs.input_ids,
                    max_length=len(inputs.input_ids[0]) + 512,
                    min_length=len(inputs.input_ids[0]) + 50,
                    do_sample=True,
                    top_k=50,
                    top_p=0.85,
                    temperature=0.5,
                    num_return_sequences=1,
                    no_repeat_ngram_size=3
                )
            
            answer = self.tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # Post-process answer
        answer = self._post_process_answer(answer)
        
        return answer

    def _post_process_answer(self, answer):
        """Clean up the generated answer"""
        # Remove any "Answer:" prefix that might have been generated
        answer = re.sub(r'^Answer:\s*', '', answer)
        
        # Ensure the answer ends with proper punctuation
        if answer and not answer[-1] in ['.', '!', '?']:
            answer += '.'
            
        # Remove any references or citations that might have been hallucinated
        answer = re.sub(r'\[\d+\]', '', answer)
        
        return answer.strip()
