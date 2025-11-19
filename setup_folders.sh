#!/bin/bash
# Setup folder structure for RAG pipeline
# Run: bash setup_folders.sh [customer_id]

CUSTOMER_ID="${1:-1}"

echo "Creating folder structure for: $CUSTOMER_ID"

# Base directories
BASE_DIR="./data/$CUSTOMER_ID"
RAW_DIR="$BASE_DIR/raw"
PROCESSED_DIR="$BASE_DIR/processed"
LOGS_DIR="./logs"

# Create folders
mkdir -p "$RAW_DIR"
mkdir -p "$PROCESSED_DIR"
mkdir -p "$LOGS_DIR"

echo ""
echo "✅ Folder structure created:"
echo "   📁 $RAW_DIR (put your PDFs/DOCX/emails here)"
echo "   📁 $PROCESSED_DIR (cleaned data stored here)"
echo "   📁 $LOGS_DIR (pipeline logs)"

# Create sample README in raw folder
README_PATH="$RAW_DIR/README.txt"
cat > "$README_PATH" << 'EOF'
RAW DOCUMENTS FOLDER
====================

Put your original documents here:
- PDFs
- Word documents (.docx, .doc)
- Excel files (.xlsx, .xls)
- Emails (.eml, .msg)
- HTML files
- Text files

After adding files, run:
  python add_documents.py --scan_raw

Or for full rebuild:
  python step1_clean_data.py --config config.yaml
EOF

echo ""
echo "📄 Created README in raw folder"
echo ""
echo "Next steps:"
echo "1. Copy your documents to: $RAW_DIR"
echo "2. Run: python step1_clean_data.py --config config.yaml"
echo "3. Run: python step3_create_embeddings_qdrant.py"
echo "4. Test: python query_engine.py"
