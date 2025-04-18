"""
Utility functions for translating text to different languages
using MarianMT models from Hugging Face.
"""
from transformers import MarianMTModel, MarianTokenizer
import torch

class TextTranslator:
    """
    A class to handle text translation using pre-trained MarianMT models.
    """
    # Dictionary mapping language codes to model names
    LANGUAGE_MODELS = {
        'hindi': 'Helsinki-NLP/opus-mt-en-hi',
        'spanish': 'Helsinki-NLP/opus-mt-en-es',
        'french': 'Helsinki-NLP/opus-mt-en-fr',
        'german': 'Helsinki-NLP/opus-mt-en-de',
        'chinese': 'Helsinki-NLP/opus-mt-en-zh',
        'russian': 'Helsinki-NLP/opus-mt-en-ru',
        'arabic': 'Helsinki-NLP/opus-mt-en-ar',
        'japanese': 'Helsinki-NLP/opus-mt-en-jap',
    }
    
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        # Models will be loaded on-demand to save memory
    
    def _load_model(self, target_language):
        """Load model and tokenizer for a specific language if not already loaded."""
        if target_language not in self.LANGUAGE_MODELS:
            raise ValueError(f"Unsupported language: {target_language}. Supported languages are: {', '.join(self.LANGUAGE_MODELS.keys())}")
        
        model_name = self.LANGUAGE_MODELS[target_language]
        
        if target_language not in self.models:
            print(f"Loading translation model for {target_language}...")
            self.tokenizers[target_language] = MarianTokenizer.from_pretrained(model_name)
            self.models[target_language] = MarianMTModel.from_pretrained(model_name)
            
            # Move model to GPU if available
            if torch.cuda.is_available():
                self.models[target_language].to('cuda')
                
            print(f"Model for {target_language} loaded successfully.")
    
    def translate(self, text, target_language):
        """
        Translate the given text to the target language.
        
        Args:
            text (str): The text to translate
            target_language (str): The target language (e.g., 'hindi', 'spanish')
            
        Returns:
            str: The translated text
        """
        # Load model if not already loaded
        self._load_model(target_language.lower())
        
        # Get the model and tokenizer
        model = self.models[target_language.lower()]
        tokenizer = self.tokenizers[target_language.lower()]
        
        # Split text into manageable chunks (MarianMT has input length limitations)
        max_length = 512
        chunks = self._split_text(text, max_length)
        translated_chunks = []
        
        # Process each chunk
        for chunk in chunks:
            if len(chunk) < 100:
                translated_chunks.append(chunk)
            else:
                # Tokenize
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                encoded = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
                
                # Translate
                with torch.no_grad():
                    output = model.generate(**encoded)
                
                # Decode
                translated_chunk = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
                translated_chunks.append(translated_chunk)
        
        # Join the translated chunks
        return " ".join(translated_chunks)
    
    def _split_text(self, text, max_length=512):
        """Split text into chunks that can be processed by the model."""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            # Approximate token length as word length
            word_length = len(word) + 1  # +1 for space
            
            if current_length + word_length > max_length:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_length = word_length
            else:
                current_chunk.append(word)
                current_length += word_length
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def get_available_languages(self):
        """Return a list of available target languages."""
        return list(self.LANGUAGE_MODELS.keys())

# Create a singleton instance
translator = TextTranslator()
