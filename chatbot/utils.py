# import os
# import fitz  # PyMuPDF
# import faiss
# import numpy as np
# from PIL import Image
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from transformers import DonutProcessor, VisionEncoderDecoderModel
# import torch

# class PDFDocumentChatbot:
#     def __init__(self, pdf_path,
#                  embedding_model_name="BAAI/bge-small-en-v1.5",
#                  donut_model_name="naver-clova-ix/donut-base-finetuned-docvqa",
#                  chunk_size=800, chunk_overlap=30, top_k=10):

#         self.pdf_path = pdf_path
#         self.embedding_model_name = embedding_model_name
#         self.donut_model_name = donut_model_name
#         self.chunk_size = chunk_size
#         self.chunk_overlap = chunk_overlap
#         self.top_k = top_k

#         # Load Donut processor and model
#         self.processor = DonutProcessor.from_pretrained(self.donut_model_name)
#         self.model = VisionEncoderDecoderModel.from_pretrained(self.donut_model_name).to("cpu")

#         # Extract text & create chunks
#         self.raw_text = self.extract_text_from_pdf()
#         self.chunks_with_metadata = self.create_text_chunks()

#         # Create FAISS index for text retrieval
#         self.embedding_model = HuggingFaceEmbeddings(model_name=self.embedding_model_name)
#         self.index = self.create_faiss_index()

#     def extract_text_from_pdf(self):
#         text = ""
#         doc = fitz.open(self.pdf_path)
#         for page in doc:
#             text += page.get_text("text") + "\n"
#         return text.strip()

#     def create_text_chunks(self):
#         splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
#         return splitter.split_text(self.raw_text)

#     def create_faiss_index(self):
#         embeddings = np.array([self.embedding_model.embed_query(t) for t in self.chunks_with_metadata], dtype=np.float32)
#         dim = embeddings.shape[1]
#         index = faiss.IndexFlatIP(dim)
#         index.add(embeddings)
#         return index

#     def retrieve_relevant_chunks(self, query):
#         query_vector = np.array([self.embedding_model.embed_query(query)], dtype=np.float32)
#         distances, indices = self.index.search(query_vector, self.top_k)
#         chunks = [self.chunks_with_metadata[i] for i in indices[0] if i < len(self.chunks_with_metadata)]
#         return " ".join(chunks)

#     def generate_answer(self, query, page_number=0):
#         """Use Donut to generate answer for a specific PDF page (image-based QA)."""
#         doc = fitz.open(self.pdf_path)

#         if page_number >= len(doc):
#             return f"PDF has only {len(doc)} pages. Page {page_number} is out of range."

#         # Render PDF page as image
#         page = doc[page_number]
#         pix = page.get_pixmap(dpi=200)
#         image_path = "temp_page.png"
#         pix.save(image_path)
#         image = Image.open(image_path).convert("RGB")

#         # Format the input for Donut
#         question = query.strip().rstrip("?") + "?"
#         task_prompt = f"<s_docvqa><s_question>{question}</s_question><s_answer>"

#         inputs = self.processor(images=image, text=task_prompt, return_tensors="pt").to("cpu")

#         with torch.no_grad():
#             outputs = self.model.generate(**inputs, max_length=512)

#         answer = self.processor.decode(outputs[0], skip_special_tokens=True)
#         return answer.replace(task_prompt, "").strip()




















# # import os
# # import fitz  # PyMuPDF
# # import faiss
# # import numpy as np
# # from langchain.embeddings import HuggingFaceEmbeddings
# # from langchain.text_splitter import RecursiveCharacterTextSplitter
# # from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# # import torch

# # class PDFDocumentChatbot:
# #     def __init__(self, pdf_path,
# #                  embedding_model_name="BAAI/bge-small-en-v1.5",  # Smaller & more efficient
# #                  llm_model_name="google/flan-t5-large",


# #                  chunk_size=800, chunk_overlap=300, top_k=10):
# #         """Initialize chatbot with optimized settings"""
# #         self.pdf_path = pdf_path
# #         self.embedding_model_name = embedding_model_name
# #         self.llm_model_name = llm_model_name

# #         self.chunk_size = chunk_size
# #         self.chunk_overlap = chunk_overlap
# #         self.top_k = top_k

# #         # Load models (Faster + Lower Memory Usage)
# #         self.embedding_model = HuggingFaceEmbeddings(model_name=self.embedding_model_name)
# #         from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# #         # Replace flan-t5-base with flan-t5-large
# #         self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
# #         self.model = AutoModelForSeq2SeqLM.from_pretrained      ("google/flan-t5-large").to("cpu")


# #         # Extract text & create embeddings
# #         self.raw_text = self.extract_text_from_pdf()
# #         self.chunks_with_metadata = self.create_text_chunks()
# #         self.index = self.create_faiss_index()

# #     def extract_text_from_pdf(self):
# #         """Extract text from PDF"""
# #         text = ""
# #         doc = fitz.open(self.pdf_path)
# #         for page in doc:
# #             text += page.get_text("text") + "\n"
# #         return text.strip()

# #     def create_text_chunks(self):
# #         """Split text into chunks"""
# #         text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
# #         return text_splitter.split_text(self.raw_text)

# #     def create_faiss_index(self):
# #         """Create FAISS index with FlatIP for fast retrieval"""
# #         embeddings = np.array([self.embedding_model.embed_query(text) for text in self.chunks_with_metadata], dtype=np.float32)
# #         d = embeddings.shape[1]

# #         # Use FlatIP for cosine similarity (Fastest)
# #         index = faiss.IndexFlatIP(d)
# #         index.add(embeddings)
# #         return index

# #     def retrieve_relevant_chunks(self, query):
# #         query_embedding = np.array([self.embedding_model.embed_query(query)], dtype=np.float32)
# #         distances, indices = self.index.search(query_embedding, self.top_k)

# #         retrieved_chunks = [
# #             self.chunks_with_metadata[i] for i in indices[0] if i < len(self.chunks_with_metadata)
# #         ]

# #         context = " ".join(retrieved_chunks)

# #         # Optional: avoid repetitive or small contexts
# #         context = self._clean_context(context)

# #         return context[:2500]  # or increase this to 3000 if your model can handle

# #     def _clean_context(self, context):
# #         lines = context.split('\n')
# #         seen = set()
# #         cleaned = []
# #         for line in lines:
# #             line = line.strip()
# #             if line and line not in seen:
# #                 seen.add(line)
# #                 cleaned.append(line)
# #         return "\n".join(cleaned)



# #     def generate_answer(self, query):
# #         """Generate a concise answer"""
# #         context = self.retrieve_relevant_chunks(query)

# #         print("🔍 Retrieved Context:\n", context)  # Debugging retrieval

# #         print("🧠 Final Context passed to model:\n", context)

# #         if not context:
# #             return "I could not find relevant details in the document."



# #         prompt = f"""
# #         You are a helpful assistant trained to read documents       and answer user questions accurately.

# #         Only use the information provided in the context below      to answer. Do NOT make up any information.

# #         If the answer is not present in the context, reply      with:
# #         "I could not find the answer in the document."

# #         ------------------
# #         Context:
# #         {context}
# #         ------------------

# #         Question: {query}
# #         Answer:
# #         """



# #         inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to("cpu")

# #         with torch.no_grad():
# #             output_ids = self.model.generate(
# #         inputs.input_ids,
# #         max_length=300,
# #         do_sample=False,      # Deterministic decoding
# #         temperature=None,     # Don't mix sampling params
# #         top_k=None
# #     )



# #         print("📝 Answer from model:", self.tokenizer.decode(output_ids[0], skip_special_tokens=True))

# #         return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)





























# # # import os
# # # from dotenv import load_dotenv
# # # load_dotenv()

# # # import fitz  # PyMuPDF
# # # import faiss
# # # import numpy as np
# # # from langchain.embeddings import HuggingFaceEmbeddings
# # # from langchain.text_splitter import RecursiveCharacterTextSplitter
# # # from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# # # import torch
# # # import requests



# # # class PDFDocumentChatbot:
# # #     def __init__(self, pdf_path,
# # #                  embedding_model_name="BAAI/bge-small-en-v1.5",  # Smaller & more efficient
# # #                  llm_model_name="google/flan-t5-base",
# # #                  chunk_size=500, chunk_overlap=200, top_k=5):
# # #         """Initialize chatbot with optimized settings"""
# # #         self.pdf_path = pdf_path
# # #         self.embedding_model_name = embedding_model_name
# # #         self.llm_model_name = llm_model_name

# # #         self.chunk_size = chunk_size
# # #         self.chunk_overlap = chunk_overlap
# # #         self.top_k = top_k

# # #         # Load models (Faster + Lower Memory Usage)
# # #         self.embedding_model = HuggingFaceEmbeddings(model_name=self.embedding_model_name)
# # #         self.tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)
# # #         self.model = AutoModelForSeq2SeqLM.from_pretrained(self.llm_model_name).to("cpu")

# # #         # Extract text & create embeddings
# # #         self.raw_text = self.extract_text_from_pdf()
# # #         self.chunks_with_metadata = self.create_text_chunks()
# # #         self.index = self.create_faiss_index()

# # #     def extract_text_from_pdf(self):
# # #         """Extract text from PDF"""
# # #         text = ""
# # #         doc = fitz.open(self.pdf_path)
# # #         for page in doc:
# # #             text += page.get_text("text") + "\n"
# # #         return text.strip()

# # #     def create_text_chunks(self):
# # #         """Split text into chunks"""
# # #         text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
# # #         return text_splitter.split_text(self.raw_text)

# # #     def create_faiss_index(self):
# # #         """Create FAISS index with FlatIP for fast retrieval"""
# # #         embeddings = np.array([self.embedding_model.embed_query(text) for text in self.chunks_with_metadata], dtype=np.float32)
# # #         d = embeddings.shape[1]

# # #         # Use FlatIP for cosine similarity (Fastest)
# # #         index = faiss.IndexFlatIP(d)
# # #         index.add(embeddings)
# # #         return index

# # #     def retrieve_relevant_chunks(self, query):
# # #         """Retrieve top matching document chunks"""
# # #         query_embedding = np.array([self.embedding_model.embed_query(query)], dtype=np.float32)
# # #         distances, indices = self.index.search(query_embedding, self.top_k)

# # #         retrieved_chunks = [self.chunks_with_metadata[i] for i in indices[0] if i < len(self.chunks_with_metadata)]

# # #         # If no good match found
# # #         if distances[0][0] < 0.6:
# # #             return ""

# # #         return " ".join(retrieved_chunks)[:2500]  # Keeps it optimized

# # #     def generate_answer(self, query):
# # #         """Generate a concise answer using Hugging Face API"""
# # #         context = self.retrieve_relevant_chunks(query)

# # #         print("🔍 Retrieved Context:\n", context)

# # #         if not context:
# # #             return "I could not find relevant details in the document."

# # #         prompt = f"""
# # #         You are an AI assistant. Use ONLY the following document context to answer.

# # #         ### Document Context:
# # #         {context}

# # #         ### Question:
# # #         {query}

# # #         ### Answer:
# # #         """

# # #         api_url = "https://api-inference.huggingface.co/models/google/flan-t5-small"

# # #         headers = {
# # #             "Authorization": "Bearer YOUR_HUGGINGFACE_TOKEN_HERE",
# # #             "Content-Type": "application/json"
# # #         }

# # #         payload = {
# # #         "inputs": prompt,
# # #         "parameters": {
# # #         "max_new_tokens": 250,  # <= fix is here
# # #         "temperature": 0.7,
# # #         "do_sample": True,
# # #         "top_k": 30
# # #     }
# # # }


# # #         print("✅ Token being used:", os.getenv("HF_API_TOKEN"))

# # #         response = requests.post(api_url, headers=headers, json=payload)

# # #         if response.status_code == 200:
# # #             return response.json()[0]["generated_text"].split("### Answer:")[-1].strip()
# # #         else:
# # #             print("API Error:", response.text)
# # #             return "Sorry, there was an issue generating the answer."
