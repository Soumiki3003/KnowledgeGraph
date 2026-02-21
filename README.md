# KnowledgeGraph

## Prerequisites

This project requires **Poppler** to be installed on your system for PDF processing and visual content extraction.

This project also requires **Tesseract OCR** to be installed for OCR/text extraction.

### Installing Poppler

- **Windows**: Download from [poppler-windows releases](https://github.com/oschwartz10612/poppler-windows/releases) and add the `bin` folder to your system PATH
- **macOS**: `brew install poppler`
- **Linux**: `sudo apt-get install poppler-utils` (Debian/Ubuntu) or `sudo yum install poppler-utils` (RedHat/CentOS)

Verify installation by running:
```bash
pdftoppm -v
```

### Installing Tesseract OCR

- **Windows**:
	- Install `tesseract` and `tesseract-languages` via Scoop:
		- `scoop install tesseract`
		- `scoop install tesseract-languages`
	- If `tesseract-languages` fails with `Cannot create symbolic link` / `A required privilege is not held by the client`, enable **Windows Developer Mode** or run PowerShell as Administrator, then retry.
- **macOS**: `brew install tesseract`
- **Linux**: `sudo apt-get install tesseract-ocr tesseract-ocr-all` (Debian/Ubuntu)

Verify installation by running:
```bash
tesseract --version
```