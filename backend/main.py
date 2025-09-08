import os
import json
import uuid
import logging
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError

from services.workflow_service import WorkflowService
from services.knowledge_base_service import KnowledgeBaseService
from models import ItemApprovalRequest

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI(
    title="BOM Platform API",
    description="Backend API for the autonomous BOM processing platform with Gemini integration.",
    version="4.0.0",
)

# Configure CORS
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add a custom exception handler for validation errors to get detailed logs
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logging.error(f"Validation Error: {exc.errors()} for request to {request.url}")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()}
    )

@app.on_event("startup")
async def startup_event():
    """Initializes the database and creates directories on startup."""
    try:
        from models import init_db
        init_db()
        os.makedirs(workflow_service.upload_dir, exist_ok=True)
        os.makedirs(workflow_service.results_dir, exist_ok=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server startup failed: {e}")

# Initialize services
workflow_service = WorkflowService()
kb_service = KnowledgeBaseService()

@app.get("/api/workflows")
async def get_workflows():
    """Get all workflows from the database."""
    try:
        workflows = workflow_service.get_all_workflows()
        return JSONResponse(content={'success': True, 'workflows': workflows})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/workflows/{workflow_id}")
async def delete_workflow(workflow_id: str):
    """Deletes a workflow and all associated data."""
    try:
        workflow_service.delete_workflow(workflow_id)
        return JSONResponse(content={'success': True, 'message': f'Workflow {workflow_id} deleted successfully'})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/knowledge-base")
async def get_knowledge_base(search: Optional[str] = "", limit: int = 50):
    """Get knowledge base items with statistics, with optional search."""
    try:
        items = kb_service.get_items(search, limit)
        stats = kb_service.get_stats()
        return JSONResponse(content={'success': True, 'items': items, 'stats': stats})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/knowledge-base/pending")
async def get_pending_approvals():
    """Get pending items for approval."""
    try:
        pending_items = kb_service.get_pending_approvals()
        return JSONResponse(content={'success': True, 'pending_items': pending_items})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/knowledge-base/approve")
async def approve_knowledge_base_item(request: ItemApprovalRequest):
    """Approve an item for the knowledge base."""
    try:
        logging.info(f"Received approval request for items: {request.item_ids} from workflow: {request.workflow_id}")
        result = kb_service.approve_items(request.item_ids)
        return JSONResponse(content={'success': True, 'approved_count': result})
    except Exception as e:
        logging.error(f"Error approving items: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/knowledge-base/reject")
async def reject_knowledge_base_item(request: ItemApprovalRequest):
    """Reject an item from the knowledge base."""
    try:
        logging.info(f"Received rejection request for items: {request.item_ids} from workflow: {request.workflow_id}")
        result = kb_service.reject_items(request.item_ids)
        return JSONResponse(content={'success': True, 'rejected_count': result})
    except Exception as e:
        logging.error(f"Error rejecting items: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/knowledge-base/{item_id}")
async def delete_knowledge_base_item(item_id: int):
    """Delete an item from the knowledge base."""
    try:
        kb_service.delete_item(item_id)
        return JSONResponse(content={'success': True, 'message': 'Item deleted successfully'})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/autonomous/upload")
async def upload_documents(
    wi_document: UploadFile = File(..., description="The Japanese WI/QC document to process."),
    item_master: Optional[UploadFile] = File(None, description="Optional Item Master for full comparison mode."),
    comparison_mode: str = Form(..., description="'full' or 'kb_only'")
):
    """Enhanced upload endpoint with optional Item Master and Gemini processing."""
    try:
        if not wi_document:
            raise HTTPException(status_code=400, detail="WI document is required")

        if comparison_mode == 'full' and not item_master:
            raise HTTPException(status_code=400, detail="Item Master is required for full comparison mode")

        workflow_id = str(uuid.uuid4())

        # Start processing asynchronously
        workflow_service.start_workflow(
            workflow_id=workflow_id,
            wi_document=wi_document,
            item_master=item_master,
            comparison_mode=comparison_mode
        )

        return JSONResponse(content={
            'success': True,
            'workflow_id': workflow_id,
            'message': 'Processing started successfully'
        })

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start workflow: {str(e)}")

@app.get("/api/autonomous/workflow/{workflow_id}/status")
async def get_workflow_status(workflow_id: str):
    """Get workflow status."""
    try:
        status = workflow_service.get_workflow_status(workflow_id)
        return JSONResponse(content=status)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Workflow not found: {str(e)}")

@app.get("/api/autonomous/workflow/{workflow_id}/results")
async def get_workflow_results(workflow_id: str):
    """Get workflow results."""
    try:
        results = workflow_service.get_workflow_results(workflow_id)
        return JSONResponse(content=results)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Results not found: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
