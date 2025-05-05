import os
import re
import time
import fitz  # PyMuPDF
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
import pdfplumber
import pytesseract
from PIL import Image, ImageEnhance
import io
import tempfile
import subprocess
import pandas as pd
from typing import Dict, List, Optional, Tuple
import json
import openai
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    logger.error("OPENAI_API_KEY environment variable is not set")
    client = None
else:
    import openai
    openai.api_key = api_key
    client = openai

# Configure Tesseract path
try:
    # Try to find Tesseract in common locations
    tesseract_paths = [
        'C:\\Program Files\\Tesseract-OCR\\tesseract.exe',  # Windows default path
        'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe',  # Windows 32-bit path
        '/usr/local/bin/tesseract',  # New installation path
        '/usr/bin/tesseract',
        '/opt/homebrew/bin/tesseract'
    ]
    
    tesseract_found = False
    for path in tesseract_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            logger.info(f"Set Tesseract path to: {path}")
            tesseract_found = True
            break
    
    if not tesseract_found:
        # If not found in common locations, try which/where command
        try:
            import subprocess
            if os.name == 'nt':  # Windows
                try:
                    tesseract_path = subprocess.check_output(['where', 'tesseract']).decode('utf-8').strip()
                except subprocess.CalledProcessError:
                    logger.error("Tesseract not found in PATH. Please install Tesseract from https://github.com/UB-Mannheim/tesseract/wiki")
                    raise
            else:  # Unix-like
                try:
                    tesseract_path = subprocess.check_output(['which', 'tesseract']).decode('utf-8').strip()
                except subprocess.CalledProcessError:
                    logger.error("Tesseract not found in PATH. Please install Tesseract using your package manager.")
                    raise
                    
            if tesseract_path:
                pytesseract.pytesseract.tesseract_cmd = tesseract_path
                logger.info(f"Set Tesseract path to: {tesseract_path}")
        except Exception as e:
            logger.error(f"Could not find Tesseract: {e}")
            logger.error("Please install Tesseract OCR and ensure it's in your PATH")
            logger.error("Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki")
            logger.error("Linux: sudo apt-get install tesseract-ocr")
            logger.error("macOS: brew install tesseract")
            raise
            
    # Verify TESSDATA_PREFIX
    tessdata_prefix = os.getenv('TESSDATA_PREFIX')
    if not tessdata_prefix:
        # Try default Windows path
        default_windows_path = 'C:\\Program Files\\Tesseract-OCR\\tessdata'
        if os.path.exists(default_windows_path):
            os.environ['TESSDATA_PREFIX'] = default_windows_path
            tessdata_prefix = default_windows_path
            logger.info(f"Using default Windows tessdata path: {tessdata_prefix}")
        else:
            error_msg = """TESSDATA_PREFIX environment variable is not set and default path not found.
Please set TESSDATA_PREFIX to point to your tessdata directory:
1. Windows: C:\\Program Files\\Tesseract-OCR\\tessdata
2. Linux: /usr/share/tesseract-ocr/4.00/tessdata
3. macOS: /usr/local/share/tessdata"""
            logger.error(error_msg)
            raise EnvironmentError(error_msg)
    elif not os.path.exists(tessdata_prefix):
        error_msg = f"Tessdata directory not found at: {tessdata_prefix}"
        logger.error(error_msg)
        raise EnvironmentError(error_msg)
    else:
        logger.info(f"Tessdata directory found at: {tessdata_prefix}")
        
except Exception as e:
    logger.error(f"Error setting Tesseract path: {e}")
    raise

class PDFRepairEngine:
    """Class to handle PDF repair and preprocessing"""
    
    @staticmethod
    def repair_with_pikepdf(pdf_path: str) -> str:
        """Attempt to repair corrupted PDF files using pikepdf"""
        try:
            from pikepdf import Pdf
            repaired_path = pdf_path.replace('.pdf', '_repaired_pike.pdf')
            
            with Pdf.open(pdf_path, allow_overwriting_input=True) as pdf:
                pdf.save(repaired_path)
                
            return repaired_path
        except Exception as e:
            logger.error(f"PikePDF repair failed: {e}")
            return pdf_path
    
    @staticmethod
    def repair_with_ghostscript(pdf_path: str) -> str:
        """Attempt to repair PDF using Ghostscript"""
        try:
            repaired_path = pdf_path.replace('.pdf', '_repaired_gs.pdf')
            # Install ghostscript if not already installed
            try:
                subprocess.run(["which", "gs"], check=True, capture_output=True)
            except subprocess.CalledProcessError:
                subprocess.run(["sudo", "apt-get", "install", "-y", "ghostscript"], check=True)
                
            # Use Ghostscript to create a clean copy
            subprocess.run([
                "gs", 
                "-o", repaired_path,
                "-sDEVICE=pdfwrite", 
                "-dPDFSETTINGS=/prepress",
                pdf_path
            ], check=True)
            
            return repaired_path
        except Exception as e:
            logger.error(f"Ghostscript repair failed: {e}")
            return pdf_path
    
    @staticmethod
    def repair_with_mutool(pdf_path: str) -> str:
        """Attempt to repair PDF using MuTool (part of MuPDF)"""
        try:
            repaired_path = pdf_path.replace('.pdf', '_repaired_mutool.pdf')
            # Install mupdf-tools if not already installed
            try:
                subprocess.run(["which", "mutool"], check=True, capture_output=True)
            except subprocess.CalledProcessError:
                subprocess.run(["sudo", "apt-get", "install", "-y", "mupdf-tools"], check=True)
                
            # Use mutool to clean the PDF
            subprocess.run([
                "mutool", "clean",
                pdf_path,
                repaired_path
            ], check=True)
            
            return repaired_path
        except Exception as e:
            logger.error(f"MuTool repair failed: {e}")
            return pdf_path
    
    @classmethod
    def full_repair_pipeline(cls, pdf_path: str) -> List[str]:
        """Run all repair methods and return list of repaired PDFs"""
        repaired_paths = [pdf_path]  # Include original as fallback
        
        # Try all repair methods
        repair_methods = [
            cls.repair_with_pikepdf,
            cls.repair_with_ghostscript, 
            cls.repair_with_mutool
        ]
        
        for repair_method in repair_methods:
            try:
                repaired_path = repair_method(pdf_path)
                if os.path.exists(repaired_path) and os.path.getsize(repaired_path) > 0:
                    repaired_paths.append(repaired_path)
            except Exception as e:
                logger.error(f"Repair method failed: {e}")
                
        return repaired_paths

class EnhancedPDFExtractor:
    """Class for extracting text from PDFs with multiple fallback methods"""
    
    @staticmethod
    def preprocess_image(img: Image.Image) -> Image.Image:
        """Enhance image for better OCR results"""
        # Convert to grayscale
        img = img.convert('L')
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2.0)
        
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(2.0)
        
        return img
    
    @staticmethod
    def extract_text_from_image(img: Image.Image) -> str:
        """Extract text from an image using advanced OCR settings"""
        try:
            # Preprocess the image
            img = EnhancedPDFExtractor.preprocess_image(img)
            
            # Check if Tesseract is available and in PATH
            try:
                # Try to get Tesseract version
                version = pytesseract.get_tesseract_version()
                logger.info(f"Tesseract version: {version}")
                
                # Try to find Tesseract executable
                tesseract_cmd = pytesseract.pytesseract.tesseract_cmd
                if not os.path.exists(tesseract_cmd):
                    logger.error(f"Tesseract executable not found at: {tesseract_cmd}")
                    return ""
                    
                # Check if TESSDATA_PREFIX is set
                tessdata_prefix = os.getenv('TESSDATA_PREFIX')
                if not tessdata_prefix:
                    logger.error("TESSDATA_PREFIX environment variable is not set")
                    return ""
                    
                # Check if tessdata directory exists
                if not os.path.exists(tessdata_prefix):
                    logger.error(f"Tessdata directory not found at: {tessdata_prefix}")
                    return ""
                    
            except Exception as e:
                logger.error(f"Tesseract check failed: {e}")
                return ""
            
            # Run OCR with multiple configurations and take the best result
            configs = [
                '--psm 6',  # Assume single uniform block of text
                '--psm 3',  # Fully automatic page segmentation
                '--psm 4'   # Assume single column of text
            ]
            
            results = []
            for config in configs:
                try:
                    text = pytesseract.image_to_string(
                        img,
                        config = f'{config} -c tessedit_char_whitelist="0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,:;/\\-_+=()[]{{}}"')
                    results.append(text)
                except Exception as e:
                    logger.error(f"OCR failed with config {config}: {e}")
                    continue
            
            # Choose the result with the most content
            return max(results, key=len) if results else ""
        
        except Exception as e:
            logger.error(f"Image OCR error: {e}")
            return ""
    
    @staticmethod
    def extract_page_with_pdfplumber(page, page_num: int) -> str:
        """Extract text from a page using pdfplumber"""
        try:
            text = page.extract_text() or ""
            if len(text.strip()) > 10:
                return text
                
            # If text extraction fails, try OCR on the page image
            img = page.to_image(resolution=300).original
            text = EnhancedPDFExtractor.extract_text_from_image(img)
            return text
        except Exception as e:
            logger.error(f"pdfplumber error on page {page_num}: {e}")
            return ""
    
    @staticmethod
    def extract_page_with_pymupdf(doc, page_num: int) -> str:
        """Extract text from a page using PyMuPDF"""
        try:
            page = doc[page_num-1]
            
            # First try native text extraction
            text = page.get_text()
            if len(text.strip()) > 10:
                return text
                
            # If that fails, render to image and use OCR
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better OCR
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text = EnhancedPDFExtractor.extract_text_from_image(img)
            return text
        except Exception as e:
            logger.error(f"PyMuPDF error on page {page_num}: {e}")
            return ""
    
    @staticmethod
    def extract_page_with_tesseract(doc, page_num: int) -> str:
        """Extract text from a page using direct Tesseract OCR as last resort"""
        try:
            # Get page and render to image at higher resolution
            page = doc[page_num-1]
            pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))  # 3x zoom for better OCR
            
            # Save to temporary file (sometimes works better than in-memory)
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                pix.save(tmp.name)
                
            # Run OCR with multiple configurations
            img = Image.open(tmp.name)
            text = EnhancedPDFExtractor.extract_text_from_image(img)
            
            # Clean up
            os.unlink(tmp.name)
            
            return text
        except Exception as e:
            logger.error(f"Tesseract direct OCR error on page {page_num}: {e}")
            return ""
    
    @staticmethod
    def extract_page_text(page, page_num: int, pdf_path: str) -> str:
        """Extract text from a single PDF page with multiple fallback methods"""
        result_text = ""
        
        # Method 1: pdfplumber
        text = EnhancedPDFExtractor.extract_page_with_pdfplumber(page, page_num)
        if len(text.strip()) > 10:
            result_text = text
        
        # Method 2: PyMuPDF if pdfplumber fails
        if not result_text:
            try:
                doc = fitz.open(page.pdf.stream)
                text = EnhancedPDFExtractor.extract_page_with_pymupdf(doc, page_num)
                if text:
                    result_text = text
                doc.close()
            except Exception as e:
                logger.error(f"PyMuPDF fallback failed for page {page_num}: {e}")
        
        # Method 3: Direct Tesseract as last resort
        if not result_text:
            try:
                doc = fitz.open(page.pdf.stream)
                text = EnhancedPDFExtractor.extract_page_with_tesseract(doc, page_num)
                if text:
                    result_text = text
                doc.close()
            except Exception as e:
                logger.error(f"Tesseract fallback failed for page {page_num}: {e}")
        
        # If all methods fail, return error message
        if not result_text.strip():
            return f"--- PAGE {page_num} ---\n[TEXT EXTRACTION FAILED]"
        
        return f"--- PAGE {page_num} ---\n{result_text.strip()}"
    
    @staticmethod
    def extract_text_with_pdfplumber(pdf_path: str, max_workers: int = 4) -> str:
        """Extract text using pdfplumber with parallel processing"""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                # Process pages in parallel
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [
                        executor.submit(
                            EnhancedPDFExtractor.extract_page_text, 
                            page, 
                            i+1,
                            pdf_path
                        )
                        for i, page in enumerate(pdf.pages)
                    ]
                    results = [future.result() for future in futures]
                
            return "\n\n".join(results)
        except Exception as e:
            logger.error(f"pdfplumber extraction failed completely: {e}")
            return ""
    
    @staticmethod
    def extract_text_with_pymupdf(pdf_path: str) -> str:
        """Extract text using PyMuPDF as fallback"""
        try:
            doc = fitz.open(pdf_path)
            results = []
            
            for page_num, page in enumerate(doc, 1):
                # Try native text extraction first
                text = page.get_text()
                
                # If little or no text, try OCR
                if len(text.strip()) < 10:
                    try:
                        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        text = EnhancedPDFExtractor.extract_text_from_image(img)
                    except Exception as e:
                        logger.error(f"PyMuPDF page {page_num} OCR failed: {e}")
                
                results.append(f"--- PAGE {page_num} ---\n{text.strip()}")
            
            doc.close()
            return "\n\n".join(results)
        except Exception as e:
            logger.error(f"PyMuPDF extraction failed completely: {e}")
            return ""
    
    @staticmethod
    def ocr_pdf_with_tesseract(pdf_path: str) -> str:
        """Direct OCR of all PDF pages as last resort method"""
        try:
            doc = fitz.open(pdf_path)
            results = []
            
            for page_num, page in enumerate(doc, 1):
                logger.info(f"Processing page {page_num} with direct OCR...")
                pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))  # Higher resolution
                
                # Save to temporary file (sometimes works better than in-memory)
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    pix.save(tmp.name)
                    
                # Open with PIL and enhance
                img = Image.open(tmp.name)
                img = EnhancedPDFExtractor.preprocess_image(img)
                
                # OCR with multiple settings
                text = EnhancedPDFExtractor.extract_text_from_image(img)
                results.append(f"--- PAGE {page_num} ---\n{text.strip()}")
                
                # Clean up
                os.unlink(tmp.name)
            
            doc.close()
            return "\n\n".join(results)
        except Exception as e:
            logger.error(f"Direct OCR method failed completely: {e}")
            return ""
    
    @classmethod
    def extract_pdf_text(cls, pdf_path: str, max_workers: int = 1) -> str:
        """Extract text from PDF with multiple fallback methods"""
        logger.info(f"Starting text extraction from: {pdf_path}")
        
        # Try multiple methods in order of preference
        extraction_methods = [
            (cls.extract_text_with_pdfplumber, {"pdf_path": pdf_path, "max_workers": max_workers}),
            (cls.extract_text_with_pymupdf, {"pdf_path": pdf_path}),
            (cls.ocr_pdf_with_tesseract, {"pdf_path": pdf_path})
        ]
        
        for method, params in extraction_methods:
            try:
                logger.info(f"Trying extraction method: {method.__name__}")
                text = method(**params)
                
                # Check if extraction was successful
                if text and len(text.strip()) > 100:
                    logger.info(f"Successful extraction with {method.__name__}")
                    return text
                else:
                    logger.warning(f"Method {method.__name__} returned insufficient text")
            except Exception as e:
                logger.error(f"Method {method.__name__} failed: {e}")
                continue
        
        return "[PDF TEXT EXTRACTION FAILED WITH ALL METHODS]"

# Initialize OpenAI client with environment variable
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    print("Warning: OPENAI_API_KEY environment variable is not set. Some features may not work.")
    client = None
else:
    openai.api_key = api_key
    client = openai

# PDF Text Extraction with Multi-threading
def extract_page_text(page, page_num):
    """Extract text from a single PDF page with OCR fallback"""
    try:
        # Try text extraction first
        text = page.extract_text() or ""
        
        # Only attempt OCR if no meaningful text found and OCR is available
        if len(text.strip()) < 10:  # More lenient check for text quality
            try:
                # Check if tesseract is available
                pytesseract.get_tesseract_version()
                img = page.to_image(resolution=100).original  # Reduced resolution
                ocr_text = pytesseract.image_to_string(
                    img,
                    config='--psm 6 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.:/- '
                )
                if len(ocr_text.strip()) > 10:  # Only use OCR if it found meaningful text
                    text = ocr_text
            except Exception as e:
                print(f"OCR failed for page {page_num}: {str(e)}")
                # Silently continue without OCR if it's not available
                pass
        
        if not text.strip():
            print(f"Warning: No text extracted from page {page_num}")
            return f"--- PAGE {page_num} ---\n[No text content found]"
            
        return f"--- PAGE {page_num} ---\n{text.strip()}"
    except Exception as e:
        print(f"Error extracting text from page {page_num}: {str(e)}")
        return f"--- PAGE {page_num} ---\n[Error extracting text: {str(e)}]"

def extract_pdf_text(pdf_path: str, max_workers: int = 1) -> str:
    """Extract text from PDF using multi-threading with reduced memory usage"""
    try:
        text_chunks = []
        with pdfplumber.open(pdf_path) as pdf:
            if not pdf.pages:
                raise ValueError("PDF file is empty or corrupted")
                
            print(f"Processing PDF with {len(pdf.pages)} pages")
            # Process pages in smaller batches
            batch_size = 2  # Reduced batch size
            for i in range(0, len(pdf.pages), batch_size):
                batch = pdf.pages[i:i+batch_size]
                print(f"Processing batch {i//batch_size + 1} of {(len(pdf.pages) + batch_size - 1)//batch_size}")
                
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [
                        executor.submit(extract_page_text, page, i+j+1)
                        for j, page in enumerate(batch)
                    ]
                    batch_results = [future.result() for future in futures]
                    text_chunks.extend(batch_results)
                
                # Clear memory after each batch
                for page in batch:
                    page.flush_cache()
                # Force garbage collection
                import gc
                gc.collect()
        
        if not text_chunks:
            raise ValueError("No text could be extracted from the PDF")
            
        return "\n\n".join(text_chunks)
    except Exception as e:
        print(f"Error in extract_pdf_text: {str(e)}")
        raise

# Optimized AI Value Extraction
class BatchReportValueExtractor:
    def __init__(self, model: str = "gpt-4-turbo"):
        self.model = model
        self.system_prompt = """You are a medical report analysis expert tasked with extracting lab test values from medical reports.

Instructions:
1. For each test in the provided list, find its corresponding value in the report
2. Return results as a JSON object where keys are test names and values are the extracted results
3. For each test value:
   - Match test names flexibly (considering abbreviations and different formatting)
   - Extract numerical values, ranges, or categorical results as found in the text
   - Return null for tests not found in the report
   - Ignore reference ranges and comments - only extract the actual result value
   - For test names, match regardless of capitalization, spacing, or punctuation

Example output:
{
  "Hemoglobin": "14.2 g/dL",
  "WBC": "6.7 x 10^3/μL",
  "Cholesterol, Total": "185 mg/dL",
  "ALT": null
}"""

    def extract_values_batch(self, test_names: List[str], report_text: str, batch_size: int = 40) -> Dict[str, Optional[str]]:
        """Extract values for all tests in batches to handle large test lists"""
        all_results = {}

        # Process tests in batches
        for i in range(0, len(test_names), batch_size):
            batch = test_names[i:i+batch_size]
            batch_results = self._process_batch(batch, report_text)
            all_results.update(batch_results)

            # Rate limit protection between batches
            if i + batch_size < len(test_names):
                time.sleep(0.5)

        return all_results

    def _process_batch(self, test_names: List[str], report_text: str) -> Dict[str, Optional[str]]:
        """Process a batch of test names against the report text"""
        # Format test names for the prompt
        test_list = "\n".join([f"- {test}" for test in test_names])

        user_prompt = f"""Extract values for these tests from the medical report:
{test_list}

Report Text:
{report_text[:80000]}"""  # Limit context to avoid token limits

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=2000,
                response_format={"type": "json_object"}
            )

            results = json.loads(response.choices[0].message.content)
            return results
        except Exception as e:
            logger.error(f"API Error: {e}")
            # Return empty results for the batch on error
            return {test: None for test in test_names}

# Chunking for large documents
def chunk_document(text, max_chunk_size=70000):
    """Split document into chunks if it exceeds API context window"""
    if len(text) <= max_chunk_size:
        return [text]

    chunks = []
    pages = text.split("\n\n--- PAGE")

    current_chunk = pages[0]
    for page in pages[1:]:
        # Add page marker back
        page_text = f"\n\n--- PAGE{page}"

        if len(current_chunk) + len(page_text) > max_chunk_size:
            chunks.append(current_chunk)
            current_chunk = page_text
        else:
            current_chunk += page_text

    # Add the last chunk
    if current_chunk:
        chunks.append(current_chunk)

    return chunks

# Main processing function
def process_report(pdf_path: str, test_names: list, batch_size: int = 5) -> Dict[str, Optional[str]]:
    """Process PDF and extract values for all test names efficiently"""
    try:
        if not os.path.exists(pdf_path):
            raise ValueError(f"PDF file not found: {pdf_path}")
            
        if not test_names:
            raise ValueError("No test names provided")
            
        # Initialize OpenAI client
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        
        # Extract text from PDF
        logger.info(f"Extracting text from PDF: {pdf_path}")
        text = EnhancedPDFExtractor.extract_pdf_text(pdf_path, max_workers=1)
        
        if not text:
            raise ValueError("No text could be extracted from the PDF")
        
        # Process text in chunks
        chunks = chunk_document(text, max_chunk_size=15000)
        logger.info(f"Document split into {len(chunks)} chunks")
        
        # Initialize value extractor
        extractor = BatchReportValueExtractor()
        
        # Process each chunk
        all_results = {}
        for i, chunk in enumerate(chunks, 1):
            logger.info(f"Processing chunk {i}/{len(chunks)}")
            logger.info(f"Chunk size: {len(chunk)} characters")
            
            try:
                # Extract values for this chunk
                chunk_results = extractor.extract_values_batch(test_names, chunk, batch_size)
                
                # Merge results, prioritizing non-null values
                for test, value in chunk_results.items():
                    if value is not None or test not in all_results:
                        all_results[test] = value
                        
            except Exception as e:
                logger.error(f"Error processing chunk {i}: {str(e)}")
                continue

        # Fill in any missing tests
        for test in test_names:
            if test not in all_results:
                all_results[test] = None

        return all_results
    except Exception as e:
        logger.error(f"Error processing report: {str(e)}")
        raise

# Value normalization functions
def extract_numerical(value):
    """Extract numerical part from a value string"""
    if pd.isna(value) or value == 'Not Found':
        return None
    # Regex to capture numerical components including ranges and modifiers
    match = re.search(
        r'^([<>≤≥±]?[\d,]*\.?\d+[\d\-\–x×/^±]*)',
        str(value).split()[0]  # Take first part to avoid units
    )
    if match:
        cleaned = match.group(1).replace(',', '').strip()
        # Handle ranges and special characters
        if any(c in cleaned for c in ['–', '-', '±', 'x', '×', '/']):
            return cleaned
        try:
            return float(cleaned)
        except:
            return cleaned
    # Handle categorical values
    if 'negative' in str(value).lower():
        return 'Negative'
    elif 'positive' in str(value).lower():
        return 'Positive'
    return None

def clean_name_v2(name):
    """Enhanced name cleaning function"""
    if pd.isna(name):
        return None
    return (
        str(name).strip().upper()
        .translate(str.maketrans('', '', ' -_/.,'))  # Remove special chars
    )

def normalize_value(value):
    """Enhanced value normalization with unit handling"""
    if pd.isna(value) or value in ['Not Found', 'nan', 'NaN']:
        return None
    try:
        # Remove units and special characters
        cleaned = re.sub(r'[^0-9.,<>≤≥±\-–/x×^]', '', str(value))
        # Handle different decimal separators
        cleaned = cleaned.replace(',', '.')
        # Handle ranges and special characters
        if any(c in cleaned for c in ['-', '–', '/']):
            return cleaned
        return float(cleaned)
    except:
        return None

# Results validation function
def create_validation_report(true_data_path, result_df, mapping_path, output_excel_path):
    """Create validation report comparing extracted values to true values"""
    # Load data
    true_df = pd.read_csv(true_data_path)
    mapping_df = pd.read_csv(mapping_path)

    # Create mapping from standard name to Thyrocare name
    name_mapping = {}
    for _, row in mapping_df.dropna(subset=['Thyrocare Test Name']).iterrows():
        name_mapping[row['Name']] = row['Thyrocare Test Name']

    # Create reverse mapping (Thyrocare name → standard name)
    reverse_mapping = {v: k for k, v in name_mapping.items()}

    # Map result_df test names to standard names
    result_df['Standard_Name'] = result_df['Test Name'].map(reverse_mapping)

    # Merge with true data using left join (keep all true_data tests)
    merged = pd.merge(
        true_df,
        result_df[['Standard_Name', 'Value', 'Test Name']],
        left_on='Name',
        right_on='Standard_Name',
        how='left'
    )

    # Handle merged columns correctly
    merged['Extracted_Value'] = merged['Value_y'].fillna('Not Found')
    merged['True_Value'] = merged['Value_x']

    # Normalize values for comparison
    def normalize_value(value):
        if pd.isna(value) or value in ['Not Found', 'nan', 'NaN']:
            return None
        try:
            # Remove units and special characters
            cleaned = re.sub(r'[^0-9.<≥≤-]', '', str(value))
            return float(cleaned) if cleaned else None
        except:
            return str(value).strip().upper()

    merged['True_Normalized'] = merged['True_Value'].apply(normalize_value)
    merged['Extracted_Normalized'] = merged['Extracted_Value'].apply(normalize_value)

    # Create comparison
    merged['Match'] = merged.apply(
        lambda x: str(x['True_Normalized']) == str(x['Extracted_Normalized']),
        axis=1
    )

    # Select and rename final columns
    final_df = merged[[
        'Name', 'Alt Name', 'UOM',
        'True_Value', 'Extracted_Value', 'Match', 'Test Name'
    ]]

    # Save to Excel
    final_df.to_excel(output_excel_path, index=False)
    logger.info(f"Validation report saved to {output_excel_path}")
    
    return final_df 
