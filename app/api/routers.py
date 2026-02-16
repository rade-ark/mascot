"""
API routers for all MASCOT endpoints.
"""
from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas import (
    CreateEvalRunRequest,
    DocumentListResponse,
    DocumentResponse,
    ErrorResponse,
    EvalRunResponse,
    HealthResponse,
    IngestURLRequest,
    QueryRequest,
    QueryResponse,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
    TaskResponse,
)
from app.core.exceptions import (
    DocumentNotFoundError,
    IngestionError,
    MascotError,
    RetrievalError,
    ValidationError,
)
from app.db.models import DocumentStatus, get_db
from app.evaluation.service import EvalSample, EvaluationService
from app.ingestion.service import IngestionService
from app.ingestion.storage import get_storage
from app.monitoring.metrics import run_health_checks
from app.workers.tasks import process_document_task


# ── Documents ──────────────────────────────────────────────────────────────────

documents_router = APIRouter(prefix="/documents", tags=["Documents"])


@documents_router.post(
    "/upload",
    response_model=TaskResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload a document for ingestion",
)
async def upload_document(
    file: UploadFile = File(...),
    metadata: str = Form(default="{}"),
    db: AsyncSession = Depends(get_db),
):
    import json

    try:
        meta = json.loads(metadata)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid metadata JSON")

    storage = get_storage()
    svc = IngestionService(db=db, storage=storage)

    try:
        doc = await svc.ingest_file(
            file_obj=file.file,
            filename=file.filename or "upload",
            metadata=meta,
        )
    except (IngestionError, ValidationError) as e:
        raise HTTPException(status_code=422, detail=e.message)

    # Kick off async processing pipeline
    task = process_document_task.delay(str(doc.id))

    return TaskResponse(
        task_id=task.id,
        status="accepted",
        message=f"Document {doc.id} queued for processing",
    )


@documents_router.post(
    "/ingest-url",
    response_model=TaskResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Ingest a document from a URL",
)
async def ingest_url(
    request: IngestURLRequest,
    db: AsyncSession = Depends(get_db),
):
    storage = get_storage()
    svc = IngestionService(db=db, storage=storage)

    try:
        doc = await svc.ingest_url(url=str(request.url), metadata=request.metadata)
    except (IngestionError, ValidationError) as e:
        raise HTTPException(status_code=422, detail=e.message)

    task = process_document_task.delay(str(doc.id))
    return TaskResponse(
        task_id=task.id,
        status="accepted",
        message=f"Document {doc.id} queued for processing",
    )


@documents_router.get(
    "/",
    response_model=DocumentListResponse,
    summary="List all documents",
)
async def list_documents(
    status_filter: DocumentStatus | None = Query(default=None, alias="status"),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    storage = get_storage()
    svc = IngestionService(db=db, storage=storage)
    docs = await svc.list_documents(status=status_filter, limit=limit, offset=offset)
    return DocumentListResponse(
        items=[DocumentResponse.model_validate(d) for d in docs],
        total=len(docs),
        limit=limit,
        offset=offset,
    )


@documents_router.get(
    "/{document_id}",
    response_model=DocumentResponse,
    summary="Get document by ID",
)
async def get_document(
    document_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    storage = get_storage()
    svc = IngestionService(db=db, storage=storage)
    doc = await svc.get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
    return DocumentResponse.model_validate(doc)


@documents_router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a document",
)
async def delete_document(
    document_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    from sqlalchemy import delete
    from app.db.models import Document, Chunk

    doc_result = await db.execute(
        __import__("sqlalchemy", fromlist=["select"]).select(Document).where(Document.id == document_id)
    )
    doc = doc_result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
    await db.delete(doc)
    await db.flush()


# ── Retrieval ──────────────────────────────────────────────────────────────────

retrieval_router = APIRouter(prefix="/retrieval", tags=["Retrieval"])


@retrieval_router.post(
    "/search",
    response_model=SearchResponse,
    summary="Semantic search over document chunks",
)
async def search(
    request: SearchRequest,
    db: AsyncSession = Depends(get_db),
):
    from app.retrieval.service import RetrievalService

    svc = RetrievalService(db=db)
    try:
        results = await svc.search(
            query=request.query,
            top_k=request.top_k,
            score_threshold=request.score_threshold,
            document_ids=request.document_ids,
        )
    except RetrievalError as e:
        raise HTTPException(status_code=500, detail=e.message)

    return SearchResponse(
        query=request.query,
        results=[
            SearchResultItem(
                chunk_id=r.chunk_id,
                document_id=r.document_id,
                document_name=r.document_name,
                content=r.content,
                score=r.score,
                metadata=r.metadata,
            )
            for r in results
        ],
        count=len(results),
    )


@retrieval_router.post(
    "/query",
    response_model=QueryResponse,
    summary="Ask a question and get an LLM-generated answer",
)
async def query(
    request: QueryRequest,
    db: AsyncSession = Depends(get_db),
):
    from app.retrieval.service import RetrievalService
    from app.monitoring.metrics import track_query

    svc = RetrievalService(db=db)
    try:
        async with track_query():
            response = await svc.query(
                question=request.question,
                top_k=request.top_k,
                document_ids=request.document_ids,
            )
    except RetrievalError as e:
        raise HTTPException(status_code=500, detail=e.message)

    return QueryResponse(
        answer=response.answer,
        sources=[
            SearchResultItem(
                chunk_id=r.chunk_id,
                document_id=r.document_id,
                document_name=r.document_name,
                content=r.content,
                score=r.score,
                metadata=r.metadata,
            )
            for r in response.sources
        ],
        query=response.query,
        latency_ms=response.latency_ms,
        model=response.model,
    )


# ── Evaluation ─────────────────────────────────────────────────────────────────

eval_router = APIRouter(prefix="/eval", tags=["Evaluation"])


@eval_router.post(
    "/runs",
    response_model=EvalRunResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Create and run an evaluation",
)
async def create_eval_run(
    request: CreateEvalRunRequest,
    db: AsyncSession = Depends(get_db),
):
    svc = EvaluationService(db=db)
    samples = [
        EvalSample(
            question=s.question,
            expected_answer=s.expected_answer,
            document_ids=s.document_ids,
        )
        for s in request.samples
    ]
    run = await svc.run_evaluation(
        name=request.name,
        samples=samples,
        config=request.config,
    )
    return EvalRunResponse.model_validate(run)


@eval_router.get(
    "/runs/{run_id}",
    response_model=EvalRunResponse,
    summary="Get evaluation run by ID",
)
async def get_eval_run(run_id: UUID, db: AsyncSession = Depends(get_db)):
    from sqlalchemy import select
    from app.db.models import EvalRun

    result = await db.execute(select(EvalRun).where(EvalRun.id == run_id))
    run = result.scalar_one_or_none()
    if not run:
        raise HTTPException(status_code=404, detail=f"Eval run {run_id} not found")
    return EvalRunResponse.model_validate(run)


# ── Health & Metrics ───────────────────────────────────────────────────────────

health_router = APIRouter(tags=["Health"])


@health_router.get("/health", response_model=HealthResponse, summary="Health check")
async def health(db: AsyncSession = Depends(get_db)):
    status_obj = await run_health_checks(db)
    code = 200 if status_obj.healthy else 503
    return HealthResponse(healthy=status_obj.healthy, checks=status_obj.checks)


@health_router.get("/metrics", summary="Prometheus metrics")
async def metrics():
    from fastapi.responses import Response
    from app.monitoring.metrics import get_metrics_response

    data, content_type = get_metrics_response()
    return Response(content=data, media_type=content_type)