from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import shutil
import uuid

from chat import chat_manager, UPLOADS_DIR  

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

Path(UPLOADS_DIR).mkdir(parents=True, exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def index():
    try:
        with open("templates/index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return HTMLResponse("<h2>Frontend not found. Please add templates/index.html</h2>")

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    file_path = Path(UPLOADS_DIR) / f"{file_id}_{file.filename}"
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    success = chat_manager.add_file(str(file_path))
    if success:
        return {"filename": file.filename, "message": "File uploaded and indexed successfully."}
    else:
        return JSONResponse({"filename": file.filename, "message": "Failed to process file."}, status_code=400)

@app.post("/chat")
async def chat(query: str = Form(...), session_id: str = Form(None)):
    answer, context = chat_manager.generate_answer(session_id, query)
    if not session_id:
        session_id = str(uuid.uuid4())  
    return JSONResponse({
        "answer": answer,
        "session_id": session_id,
        "context": context  
    })
    

@app.post("/clear_session")
async def clear_session(session_id: str = Form(...)):
    chat_manager.clear_session(session_id)
    return {"message": f"Session {session_id} cleared."}

@app.post("/reset_all")
async def reset_all():
    chat_manager.clear_everything()
    return {"message": "All uploads, data, FAISS index, and resume data cleared."}
