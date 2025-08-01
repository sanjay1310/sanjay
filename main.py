from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from bert_score import score as bert_score

from transformers import AutoModel, AutoTokenizer, MBartForConditionalGeneration, MBartTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import numpy as np
import torch
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from transformers import AlbertTokenizer
from indicnlp.transliterate.unicode_transliterate import UnicodeIndicTransliterator
from torch.nn.functional import cosine_similarity
import PyPDF2
import docx
import io

# Global model initialization - do this only once when app starts
print("Initializing models...")
try:
    # Load IndicBART with correct model classes
    model_name = "ai4bharat/IndicBART"
    indicbart_tokenizer = AlbertTokenizer.from_pretrained(model_name, do_lower_case=False, 
                                                         use_fast=False, keep_accents=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device set to use {device}")
    indicbart_model = MBartForConditionalGeneration.from_pretrained(model_name).to(device)
    
    # Load mT5 for abstractive summarization
    mt5_model_name = "csebuetnlp/mT5_multilingual_XLSum"
    mt5_model = AutoModelForSeq2SeqLM.from_pretrained(mt5_model_name).to(device)
    mt5_tokenizer = AutoTokenizer.from_pretrained(mt5_model_name, use_fast=False, legacy=True)
    
except Exception as e:
    print(f"Error loading models: {e}")

# Download NLTK data once at startup
nltk.download('punkt', quiet=True)

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
        if lang.lower() == "telugu":
            processed_sentences = [transliterate_if_telugu(sent) for sent in sentences]
        else:
            processed_sentences = sentences
        
        # Add language tokens and EOS token
        lang_code = "te" if lang.lower() == "telugu" else "hi" if lang.lower() == "hindi" else "en"
        tokenized_sentences = [f"{sent} </s> <2{lang_code}>" for sent in processed_sentences]
        
        # Tokenize
        inputs = indicbart_tokenizer(
            tokenized_sentences, 
            add_special_tokens=False,
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=512
        )
        
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = indicbart_model.get_encoder()(input_ids=inputs["input_ids"])
            embeddings = outputs.last_hidden_state
            sentence_embeddings = embeddings.mean(dim=1)
            
        return sentence_embeddings, processed_sentences
    
    except Exception as e:
        print(f"Error in generating embeddings: {str(e)}")
        return None, None

async def generate_extractive_summary(text, language):
    try:
        # Clean input text
        text = ' '.join(text.split())
        
        # Split into sentences and clean them
        sentences = [s.strip() for s in sent_tokenize(text) if s.strip()]
        if len(sentences) <= 3:
            return text, None
        
        # Generate embeddings
        sentence_embeddings, processed_sentences = get_sentence_embeddings(sentences, language)
        if sentence_embeddings is None:
            return ' '.join(sentences[:3]), None
        
        # Calculate similarity scores
        doc_embedding = sentence_embeddings.mean(dim=0)
        similarity_matrix = cosine_similarity(sentence_embeddings, doc_embedding.unsqueeze(0))
        
        # Select top sentences (about 30% of original text)
        num_sentences = max(3, int(len(sentences) * 0.3))
        scores = similarity_matrix.squeeze()
        top_indices = scores.topk(num_sentences).indices.cpu().tolist()
        
        # Get summary sentences in original order
        summary_sentences = [sentences[idx] for idx in sorted(top_indices)]
        
        # Handle Telugu back-transliteration if needed
        if language.lower() == "telugu":
            summary_sentences = [transliterate_back_to_telugu(sent) for sent in summary_sentences]
        
        summary = ' '.join(summary_sentences)
        return summary, None
        
    except Exception as e:
        print(f"Error in generating summary: {str(e)}")
        raise HTTPException(status_code=500, 
                          detail=f"Extractive summarization failed: {str(e)}")

async def generate_abstractive_summary(text, language):
    try:
        # Use the globally loaded models instead of loading them again
        inputs = mt5_tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = mt5_model.generate(
                **inputs,
                max_length=150,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True
            )
            
        summary = mt5_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary
        
    except Exception as e:
        raise HTTPException(status_code=500,
                          detail=f"Abstractive summarization failed: {str(e)}")

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=[""],  # Allow all origins; replace "" with specific domains in production
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Mount the "static" directory to serve HTML, CSS, and JS
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("csebuetnlp/mT5_multilingual_XLSum", use_fast=False)
model = AutoModelForSeq2SeqLM.from_pretrained("csebuetnlp/mT5_multilingual_XLSum")
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, device=0)

# Request model
class SummarizationRequest(BaseModel):
    language: str
    article: str
    reference_summary: str = ""
    summary_type: str

@app.post("/summarize")
async def summarize(request: SummarizationRequest):
    try:
        if request.summary_type == "extractive":
            summary, bert_scores = await generate_extractive_summary(
                request.article, 
                request.language
            )
        else:
            summary = await generate_abstractive_summary(
                request.article, 
                request.language
            )
            bert_scores = None
            
        # Calculate BERT Score if reference summary is provided
        if request.reference_summary:
            from bert_score import score
            P, R, F1 = score([summary], [request.reference_summary], 
                           lang=request.language.lower())
            bert_scores = {
                "P": float(P.mean()),
                "R": float(R.mean()),
                "F1": float(F1.mean())
            }
        
        return {
            "summary": summary,
            "bert_score": bert_scores
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, 
                          detail=str(e))

@app.get("/")
async def read_root():
    return FileResponse('static/index.html')

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        print(f"Received file: {file.filename}")  # Debug log
        content = await file.read()
        text = ""
        
        if file.filename.endswith('.txt'):
            print("Processing txt file")  # Debug log
            text = content.decode('utf-8')
        
        elif file.filename.endswith('.pdf'):
            print("Processing pdf file")  # Debug log
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
                
        elif file.filename.endswith('.docx'):
            print("Processing docx file")  # Debug log
            doc = docx.Document(io.BytesIO(content))
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            
        elif file.filename.endswith('.doc'):
            print("Processing doc file")  # Debug log
            raise HTTPException(
                status_code=400, 
                detail="Legacy .doc files are not supported. Please convert to .docx format."
            )
            
        else:
            print(f"Unsupported file format: {file.filename}")  # Debug log
            raise HTTPException(
                status_code=400, 
                detail="Unsupported file format. Please upload .txt, .pdf, or .docx files only."
            )
            
        if not text.strip():
            raise HTTPException(
                status_code=400,
                detail="No text could be extracted from the file."
            )
            
        return {"text": text}
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error processing file: {str(e)}")  # Debug log
        import traceback
        print(traceback.format_exc())  # Print full error traceback
        raise HTTPException(
            status_code=500,
            detail="Error processing file. Please ensure the file is not corrupted and try again."
        )