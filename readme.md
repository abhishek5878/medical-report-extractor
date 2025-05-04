# Medical Report Value Extractor

A web application that extracts lab test values from medical PDF reports using OCR and AI.

## Features

- PDF upload interface
- Text extraction with OCR fallback
- AI-powered value extraction using OpenAI
- Interactive web interface
- Export results to Excel

## Prerequisites

- Python 3.8+
- Tesseract OCR
- OpenAI API key

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/medical-report-extractor.git
cd medical-report-extractor
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

4. Install Tesseract OCR:

- **Linux**: `sudo apt install tesseract-ocr`
- **macOS**: `brew install tesseract`
- **Windows**: [Download installer](https://github.com/UB-Mannheim/tesseract/wiki)

5. Create an environment file:

```bash
cp .env.example .env
```

6. Update `.env` with your OpenAI API key.

7. Create data directory and add required files:

```bash
mkdir -p data
```

8. Add the mapping file (`marker_mapping.csv`) and true data file (`true_data.csv`) to the data directory.

## File Structure

The application requires two CSV files in the `data` directory:

1. `marker_mapping.csv`: Maps standard test names to Thyrocare test names.
   - Required columns: `Name`, `Alt Name`, `Thyrocare Test Name`

2. `true_data.csv`: Contains expected values for validation.
   - Required columns: `Name`, `Alt Name`, `UOM`, `Value`

## Running the Application

Start the Flask development server:

```bash
flask run
```

Then open your browser and navigate to [http://localhost:5000](http://localhost:5000).

## Docker Support

Build and run with Docker:

```bash
docker build -t medical-report-extractor .
docker run -p 5000:5000 -e OPENAI_API_KEY=your_key_here medical-report-extractor
```

## How It Works

1. User uploads a PDF medical report
2. The application extracts text from the PDF (with OCR fallback)
3. AI extracts lab test values based on a predefined list
4. Results are formatted into an Excel file that highlights matches and mismatches
5. User downloads the Excel file with extracted values

## License

MIT
