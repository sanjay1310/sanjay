from transformers import AlbertTokenizer, MBartForConditionalGeneration
from nltk.tokenize import sent_tokenize
from torch.nn.functional import cosine_similarity
from indicnlp.transliterate.unicode_transliterate import UnicodeIndicTransliterator
import torch
import nltk
nltk.download('punkt')

print("Loading IndicBART model and tokenizer...")

# Initialize model and tokenizer with correct classes
MODEL_NAME = "ai4bharat/IndicBART"
tokenizer = AlbertTokenizer.from_pretrained(MODEL_NAME, do_lower_case=False, use_fast=False, keep_accents=True)
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
model = MBartForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)

# Get special token IDs
bos_id = tokenizer._convert_token_to_id_with_added_voc("<s>")
eos_id = tokenizer._convert_token_to_id_with_added_voc("</s>")
pad_id = tokenizer._convert_token_to_id_with_added_voc("<pad>")

def transliterate_if_telugu(text):
    """Transliterate Telugu text to Devanagari"""
    try:
        return UnicodeIndicTransliterator.transliterate(text, "te", "hi")
    except Exception as e:
        print(f"Transliteration error: {str(e)}")
        return text

def transliterate_back_to_telugu(text):
    """Transliterate Devanagari text back to Telugu"""
    try:
        return UnicodeIndicTransliterator.transliterate(text, "hi", "te")
    except Exception as e:
        print(f"Back-transliteration error: {str(e)}")
        return text

def get_sentence_embeddings(sentences, lang):
    """Generate embeddings using IndicBART with proper language tokens"""
    try:
        # Handle Telugu transliteration
        if lang == "Telugu":
            processed_sentences = [transliterate_if_telugu(sent) for sent in sentences]
        else:
            processed_sentences = sentences
        
        # Add language tokens and EOS token
        lang_code = "te" if lang == "Telugu" else "hi" if lang == "Hindi" else "en"
        tokenized_sentences = [f"{sent} </s> <2{lang_code}>" for sent in processed_sentences]
        
        # Tokenize
        inputs = tokenizer(tokenized_sentences, 
                         add_special_tokens=False,
                         return_tensors="pt", 
                         padding=True, 
                         truncation=True,
                         max_length=512)
        
        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.get_encoder()(input_ids=inputs["input_ids"])
            embeddings = outputs.last_hidden_state
            sentence_embeddings = embeddings.mean(dim=1)
            
        return sentence_embeddings, processed_sentences
    
    except Exception as e:
        print(f"Error in generating embeddings: {str(e)}")
        return None, None

def extractive_summary(text, lang, num_sentences=3):
    """Generate extractive summary using IndicBART embeddings"""
    try:
        # Clean input text by removing extra whitespace
        text = ' '.join(text.split())
        
        # Split into sentences and clean them
        sentences = [s.strip() for s in sent_tokenize(text) if s.strip()]
        if len(sentences) <= num_sentences:
            return ' '.join(sentences)
        
        # Generate embeddings
        sentence_embeddings, processed_sentences = get_sentence_embeddings(sentences, lang)
        if sentence_embeddings is None:
            return ' '.join(sentences[:num_sentences])
        
        # Calculate similarity scores
        doc_embedding = sentence_embeddings.mean(dim=0)
        similarity_matrix = cosine_similarity(sentence_embeddings, doc_embedding.unsqueeze(0))
        
        # Select top sentences
        scores = similarity_matrix.squeeze()
        top_indices = scores.topk(num_sentences).indices.cpu().tolist()
        
        # Get summary sentences in original order
        summary_sentences = [sentences[idx] for idx in sorted(top_indices)]
        
        # Handle Telugu back-transliteration if needed
        if lang == "Telugu":
            summary_sentences = [transliterate_back_to_telugu(sent) for sent in summary_sentences]
        
        # Join sentences with proper spacing
        return ' '.join(summary_sentences)
    
    except Exception as e:
        print(f"Error in generating summary: {str(e)}")
        return ' '.join(sentences[:num_sentences])

if __name__ == "__main__":
    # Test examples with cleaner formatting
    test_texts = {
        "Telugu": """తెలంగాణ రాష్ట్రం భారతదేశంలోని 29 రాష్ట్రాలలో ఒకటి. 
        హైదరాబాద్ దీని రాజధాని. 
        2014 జూన్ 2న ఆంధ్రప్రదేశ్ నుండి విడిపోయి ప్రత్యేక రాష్ట్రంగా ఏర్పడింది.
        తెలంగాణ భారతదేశంలోని దక్షిణ మధ్య ప్రాంతంలో ఉంది.""",
        
        "Hindi": """भारत एक विशाल देश है। 
        यहाँ विभिन्न धर्म, भाषा और संस्कृति के लोग रहते हैं। 
        भारत की राजधानी नई दिल्ली है। 
        यह दुनिया का सबसे बड़ा लोकतंत्र है।""",
        
        "English": """India is a vast country. 
        People of different religions, languages, and cultures live here. 
        New Delhi is the capital of India. 
        It is the world's largest democracy."""
    }
    
    for lang, text in test_texts.items():
        print(f"\n{'='*50}")
        print(f"Testing {lang} summarization:")
        print(f"{'='*50}")
        
        # Clean the input text
        text = ' '.join(text.split())
        summary = extractive_summary(text, lang, num_sentences=3)
        
        print("\nOriginal Text:")
        print(text)
        print("\nGenerated Summary:")
        print(summary)
