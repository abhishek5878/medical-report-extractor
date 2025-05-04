import os
import re
import time
import json
import pdfplumber
import pytesseract
import pandas as pd
from PIL import Image
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import openai

# Initialize OpenAI client with environment variable
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    print("Warning: OPENAI_API_KEY environment variable is not set. Some features may not work.")
    client = None
else:
    client = openai.OpenAI(api_key=api_key)

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
    def __init__(self, model: str = "gpt-4o"):
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

    def extract_values_batch(self, test_names: List[str], report_text: str, batch_size: int = 50) -> Dict[str, Optional[str]]:
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
{report_text[:80000]}"""  # Limit text to avoid token limits

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
            print(f"API Error: {e}")
            # Return empty results for the batch on error
            return {test: None for test in test_names}

# Chunking for large documents
def chunk_document(text: str, max_chunk_size: int = 70000) -> List[str]:
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
def process_report(pdf_path: str, test_names: list, batch_size: int = 10) -> Dict[str, Optional[str]]:
    """Process PDF and extract values for all test names efficiently"""
    try:
        if not os.path.exists(pdf_path):
            raise ValueError(f"PDF file not found: {pdf_path}")
            
        if not test_names:
            raise ValueError("No test names provided")
            
        # Extract text with reduced memory usage
        print(f"Extracting text from PDF: {pdf_path}")
        start_time = time.time()
        report_text = extract_pdf_text(pdf_path, max_workers=1)  # Reduced workers
        print(f"Text extraction completed in {time.time() - start_time:.2f} seconds")
        print(f"Extracted text length: {len(report_text)} characters")

        if not report_text.strip():
            raise ValueError("No text content could be extracted from the PDF")

        # Initialize extractor with smaller batch size
        extractor = BatchReportValueExtractor()
        all_results = {}

        # Handle large documents with smaller chunks
        text_chunks = chunk_document(report_text, max_chunk_size=15000)  # Further reduced chunk size
        print(f"Document split into {len(text_chunks)} chunks")

        # Process each chunk separately to avoid context limits
        for i, chunk in enumerate(text_chunks):
            print(f"Processing chunk {i+1}/{len(text_chunks)}")
            print(f"Chunk size: {len(chunk)} characters")

            # Extract values from this chunk with smaller batch size
            chunk_start = time.time()
            chunk_results = extractor.extract_values_batch(test_names, chunk, batch_size)
            print(f"Chunk {i+1} processed in {time.time() - chunk_start:.2f} seconds")

            # Merge results, prioritizing non-null values
            for test, value in chunk_results.items():
                if value is not None or test not in all_results:
                    all_results[test] = value

            # Force garbage collection after each chunk
            import gc
            gc.collect()

        # Fill in any missing tests
        for test in test_names:
            if test not in all_results:
                all_results[test] = None

        return all_results
    except Exception as e:
        print(f"Error processing report: {str(e)}")
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
    """Create Excel validation report comparing extracted values with true values"""
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
    merged['True_Value'] = merged['Value_x']  # Original true values

    # Normalize values for comparison
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

    # Create styled Excel output
    with pd.ExcelWriter(output_excel_path, engine='xlsxwriter') as writer:
        final_df.to_excel(writer, sheet_name='Validation Report', index=False)
        
        # Get workbook and worksheet
        workbook = writer.book
        worksheet = writer.sheets['Validation Report']
        
        # Define formats
        match_format = workbook.add_format({'bg_color': '#c6efce'})
        mismatch_format = workbook.add_format({'bg_color': '#ffc7ce'})
        
        # Apply conditional formatting
        for row in range(1, len(final_df) + 1):
            match_value = final_df.iloc[row-1]['Match']
            if match_value:
                worksheet.set_row(row, None, match_format)
            else:
                worksheet.set_row(row, None, mismatch_format)

        # Auto-adjust columns
        for idx, col in enumerate(final_df.columns):
            max_len = max((
                final_df[col].astype(str).map(len).max(),
                len(str(col))
            )) + 2
            worksheet.set_column(idx, idx, max_len)

    print(f"Validation report saved to {output_excel_path}")
    return output_excel_path 