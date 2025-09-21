# Plan: RCP Data Extraction Bot

This document outlines the plan for creating a robust system to download approximately 31,000 multi-page PDF documents (RCPs) and extract their textual content for AI model training.

## Phase 1: Discovery and Scoping

**Objective:** Identify the source of the RCP documents and understand the website's structure for scraping.

1.  **Identify the Source:** The primary task is to locate the official repository or database where the RCPs are hosted (e.g., European Medicines Agency - EMA, a national health authority website).
2.  **Analyze Website Structure:** Manually inspect the website to understand how the documents are listed.
    *   Is there a searchable database?
    *   Are the links listed on static pages?
    *   Is there a pattern in the URLs for the PDF files?
    *   What are the rate-limiting or anti-bot measures in place?
3.  **Define Scope:** Create a list of all entry-point URLs from which the scraper will start its work.

## Phase 2: The Scraper/Downloader Bot

**Objective:** Develop a script to systematically download all 31,000 PDF files.

**Technology Stack:**
*   **Programming Language:** Python
*   **Core Libraries:**
    *   `requests`: For making HTTP requests to download the files.
    *   `BeautifulSoup` or `lxml`: For parsing HTML and finding the links to the PDFs.
    *   `asyncio` / `aiohttp`: To perform downloads asynchronously for significant speed improvements.

**Development Steps:**

1.  **URL Crawler:**
    *   The bot will start from the scoped URLs (from Phase 1).
    *   It will parse the HTML to find all links that point to PDF files matching the RCP format.
    *   It will handle pagination if the documents are spread across multiple pages.
2.  **PDF Downloader:**
    *   For each PDF link found, the bot will download the file.
    *   Files should be saved with a systematic naming convention (e.g., `drug_name_id.pdf` or using a unique identifier from the source).
3.  **Robustness and Error Handling:**
    *   **Logging:** Implement comprehensive logging to track which files have been downloaded successfully and which have failed.
    *   **Checkpointing:** The script must be able to be stopped and restarted without re-downloading existing files. A simple check (`if os.path.exists(file_path): continue`) or a more robust database of downloaded URLs can be used.
    *   **Rate Limiting:** Implement delays (`time.sleep()`) between requests to avoid being blocked by the server. Use `aiohttp`'s client session management for handling connections gracefully.
    *   **User-Agent:** Set a proper User-Agent string to identify the bot.

## Phase 3: Text Extraction and Structuring

**Objective:** Extract clean, usable text from all downloaded PDFs and structure it.

**Technology Stack:**
*   **Programming Language:** Python
*   **Core Libraries:**
    *   `pdfplumber` or `PyMuPDF (fitz)`: These are highly effective modern libraries for extracting text, tables, and metadata from PDF files. They are generally more accurate than older libraries like `PyPDF2`.
    *   `Tesseract (pytesseract)`: To be used as a fallback for any PDFs that are scanned images rather than text-based.

**Development Steps:**

1.  **PDF Processing Pipeline:**
    *   Create a script that iterates through the directory of downloaded PDFs.
    *   For each PDF, use `pdfplumber` to open the file and extract text from every page.
2.  **Text Cleaning and Pre-processing:**
    *   Remove headers, footers, and page numbers that are common across pages.
    *   Correct common OCR errors if `Tesseract` is used.
    *   Normalize whitespace and handle hyphenation at the end of lines.
3.  **Structured Data Extraction (Crucial for the AI):**
    *   The raw text is not enough. The goal is to extract information from specific RCP sections (e.g., "4.5 Interaction with other medicinal products", "4.8 Undesirable effects", "4.6 Fertility, pregnancy and lactation").
    *   Use **Regular Expressions (RegEx)** to identify the headings of these sections and extract the content that follows until the next major heading.
4.  **Output Format:**
    *   Store the extracted, structured data in a machine-readable format. **JSON is ideal.**
    *   Create one JSON file per RCP document. The JSON structure should be a dictionary with keys corresponding to the RCP sections.

**Example JSON Output (`drug_name_id.json`):**
```json
{
  "drug_name": "ExampleDrug",
  "source_pdf": "drug_name_id.pdf",
  "sections": {
    "interactions": "Text from section 4.5...",
    "adverse_reactions": "Text from section 4.8...",
    "pregnancy": "Text from section 4.6...",
    "composition": "Text from section 2..."
  }
}
```

This structured dataset will be the foundation for training the AI model.
