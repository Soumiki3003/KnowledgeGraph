# KnowledgeGraph

## Prerequisites

This project requires **Poppler** to be installed on your system for PDF processing and visual content extraction.

### Installing Poppler

- **Windows**: Download from [poppler-windows releases](https://github.com/oschwartz10612/poppler-windows/releases) and add the `bin` folder to your system PATH
- **macOS**: `brew install poppler`
- **Linux**: `sudo apt-get install poppler-utils` (Debian/Ubuntu) or `sudo yum install poppler-utils` (RedHat/CentOS)

Verify installation by running:
```bash
pdftoppm -v
```