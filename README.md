# Resume-Based Chatbot

A smart conversational AI application that allows users to upload resumes and ask questions about the content. The chatbot uses semantic search and advanced NLP models to provide accurate answers based on resume information.

## Features

- **Multi-Format Support**: Upload resumes in PDF, DOCX, PPTX, TXT, and image formats (with OCR)
- **Semantic Search**: Uses FAISS vector database for efficient similarity search
- **Context-Aware Answers**: Generates answers based on resume context using transformer models
- **Session Management**: Maintain conversation history with unique session IDs
- **Web Interface**: User-friendly chat interface with file upload functionality
- **Real-Time Processing**: Asynchronous file processing and indexing

## Technology Stack

- **Backend**: FastAPI, Uvicorn
- **ML/NLP**: 
  - Sentence Transformers (embeddings)
  - Hugging Face Transformers (seq2seq models)
  - PyTorch
- **Vector Database**: FAISS
- **Document Processing**: PyPDF, python-docx, python-pptx, Pillow, Tesseract OCR
- **Frontend**: HTML, CSS, JavaScript

## Project Structure

```
.
├── main.py                 # FastAPI application and routes
├── chat.py                 # Chat manager and core logic
├── req.txt                 # Python dependencies
├── templates/
│   └── index.html          # Web UI template
├── static/
│   ├── css/
│   │   └── style.css       # Styling
│   ├── js/
│   │   └── app.js          # Frontend JavaScript
│   └── uploads/            # Temporary uploaded files
├── data/                   # Processed resume data
├── uploads/                # User-uploaded files
├── faiss_index/            # FAISS vector database
│   ├── index.faiss         # Vector index
│   ├── metas.json          # Metadata
│   └── texts.json          # Original texts
└── convo_store/            # Conversation history storage
```

## Installation

### Prerequisites
- Python 3.8+
- Tesseract OCR (for image processing on Windows): [Install from here](https://github.com/UB-Mannheim/tesseract/wiki)

### Setup

1. **Clone/Navigate to the project directory**:
   ```bash
   cd "resume based chatbot"
   ```

2. **Install dependencies**:
   ```bash
   pip install -r req.txt
   ```

3. **Verify Tesseract installation** (Windows):
   - Ensure Tesseract is installed at `C:\Program Files\Tesseract-OCR\tesseract.exe`
   - Or update the path in `chat.py` if installed elsewhere

## Usage

### Running the Application

1. **Start the server**:
   ```bash
   python main.py
   ```
   or
   ```bash
   uvicorn main:app --reload
   ```

2. **Access the web interface**:
   - Open your browser and go to `http://localhost:8000`

### Using the Chatbot

1. **Upload a Resume**:
   - Click on the upload box in the sidebar
   - Select a resume file (PDF, DOCX, PPTX, TXT, or image)
   - Click "Upload" to process and index the document

2. **Ask Questions**:
   - Type your question in the chat input field
   - The chatbot will search the resume and generate an answer
   - Questions are saved in the conversation history

3. **Manage Sessions**:
   - Click "New Chat" to start a fresh conversation
   - Click "Clear Chat" to remove current session data

## API Endpoints

### GET `/`
Returns the web interface HTML.

### POST `/upload`
Uploads and indexes a resume file.
- **Parameters**: `file` (UploadFile)
- **Response**: `{"filename": string, "message": string}`

### POST `/chat`
Sends a query and receives an answer.
- **Parameters**: 
  - `query` (string): The question to ask
  - `session_id` (string, optional): Session identifier
- **Response**: `{"answer": string, "session_id": string, "context": string}`

### POST `/clear_session`
Clears the conversation history for a session.
- **Parameters**: `session_id` (string)
- **Response**: `{"message": string}`

## Configuration

### Tesseract OCR Path (Windows)
Edit `chat.py` line ~27 to set the correct Tesseract path:
```python
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

### Model Selection
The chatbot uses:
- `sentence-transformers/all-MiniLM-L6-v2` for embeddings
- Hugging Face seq2seq models for answer generation

Modify these in `chat.py` to use different models.

## Dependencies

See `req.txt` for the complete list. Key dependencies:
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `sentence-transformers` - Embedding models
- `faiss-cpu` - Vector similarity search
- `transformers` - NLP models
- `torch` - Deep learning framework
- Document processing: `pypdf`, `python-docx`, `python-pptx`, `Pillow`, `pytesseract`

## Troubleshooting

### ImportError: faiss-cpu not installed
```bash
pip install faiss-cpu
```

### Tesseract not found (Windows)
- Install Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki
- Update the path in `chat.py` accordingly

### CUDA/GPU Issues
If using GPU, install:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Large Model Downloads
First run downloads embedding and LLM models. This may take time and disk space.

## Performance Tips

- Use smaller embedding models for faster processing: `all-MiniLM-L6-v2` (default)
- For faster inference, use CPU (`torch` default) or GPU if available
- FAISS index grows with each uploaded resume; periodically clean up old data

## Future Enhancements

- Multi-resume support with source tracking
- Advanced query filtering and refinement
- Conversation export/save functionality
- Custom model fine-tuning
- Real-time streaming responses
- Database persistence for chats

## License

MIT License - feel free to use and modify

## Support

For issues or questions, check the logs and ensure all dependencies are properly installed.
