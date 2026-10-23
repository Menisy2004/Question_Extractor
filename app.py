from fastapi import FastAPI, Form, Request, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool
import logging
import os
import tempfile
import uuid
from pathlib import Path
from src.helper import llm_pipeline

app = FastAPI()
BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

# Mount static folder only if it actually exists and has files
if STATIC_DIR.is_dir():
  app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

qa_sessions = {}
LOCAL_MAX_UPLOAD_BYTES = 30 * 1024 * 1024
# Vercel's function request body limit is 4.5 MB including multipart overhead.
# Keep the PDF itself below that limit so requests reach the Python function.
VERCEL_MAX_UPLOAD_BYTES = 4 * 1024 * 1024
MAX_UPLOAD_BYTES = (
    VERCEL_MAX_UPLOAD_BYTES if os.getenv("VERCEL") else LOCAL_MAX_UPLOAD_BYTES
)


@app.get("/")
async def index(request: Request):
  return templates.TemplateResponse(
      request=request,
      name="index.html",
      context={"max_upload_mb": MAX_UPLOAD_BYTES // (1024 * 1024)},
  )


@app.post("/generate")
async def generate_questions(file: UploadFile = File(...)):
    """Create a question set from an uploaded PDF and return it as JSON."""
    if not file.filename or Path(file.filename).suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Please choose a PDF file.")

    temp_path = None
    try:
        suffix = Path(file.filename).suffix.lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_path = temp_file.name
            total_bytes = 0
            while chunk := await file.read(1024 * 1024):
                total_bytes += len(chunk)
                if total_bytes > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"The PDF is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB. Please choose a smaller document.",
                    )
                temp_file.write(chunk)

        if os.path.getsize(temp_path) == 0:
            raise HTTPException(status_code=400, detail="The selected PDF is empty.")

        answer_chain, questions = await run_in_threadpool(llm_pipeline, temp_path)
        if not questions:
            raise HTTPException(
                status_code=422,
                detail="No questions were generated. Try a PDF with selectable text.",
            )

        session_id = uuid.uuid4().hex
        qa_sessions[session_id] = answer_chain
        return {"session_id": session_id, "filename": file.filename, "questions": questions}
    except HTTPException:
        raise
    except Exception as exc:
        logging.exception("Question generation failed")
        error_text = str(exc)
        if "429" in error_text and "RESOURCE_EXHAUSTED" in error_text:
            raise HTTPException(
                status_code=429,
                detail="An AI provider quota has been reached. Check the Groq and Google AI Studio usage limits, then try again.",
            ) from exc
        if "503" in error_text and "UNAVAILABLE" in error_text:
            raise HTTPException(
                status_code=503,
                detail="An AI provider is temporarily at capacity. Please try again later.",
            ) from exc
        if "403" in error_text and "PERMISSION_DENIED" in error_text:
            raise HTTPException(
                status_code=403,
                detail="Google has denied API access for this project. Check that the Google API key is valid and the Gemini API is enabled for its project.",
            ) from exc
        if any(marker in error_text.upper() for marker in ("TIMEOUT", "TIMED OUT", "DEADLINE EXCEEDED")):
            raise HTTPException(
                status_code=504,
                detail="Gemini did not respond in time. Please retry shortly or try a shorter PDF.",
            ) from exc
        raise HTTPException(
            status_code=500,
            detail="Question generation failed. Check that GOOGLE_API_KEY or GEMINI_API_KEY is configured in the deployment environment, then try again.",
        ) from exc
    finally:
        await file.close()
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/answer")
async def answer_question(session_id: str = Form(...), question: str = Form(...)):
    """Retrieve a grounded answer to one generated question."""
    answer_chain = qa_sessions.get(session_id)
    if answer_chain is None:
        raise HTTPException(status_code=404, detail="This question session has expired. Upload the PDF again.")
    if not question.strip():
        raise HTTPException(status_code=400, detail="A question is required.")

    try:
        answer = await run_in_threadpool(answer_chain.run, question.strip())
        return {"answer": answer}
    except Exception as exc:
        logging.exception("Answer retrieval failed")
        raise HTTPException(status_code=500, detail="Could not retrieve an answer. Please try again.") from exc
