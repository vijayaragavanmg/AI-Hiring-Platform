"""
Resume Pipeline — MCP Server

Exposes the resume pipeline as MCP tools so Claude (or any MCP-compatible
AI agent) can upload, search, and manage resumes via natural language.

Tools:
  health_check         — check if the API is reachable
  upload_resume        — upload a single resume (async, returns job_id)
  upload_resume_sync   — upload a single resume and wait for the result
  upload_batch         — upload multiple resumes in parallel
  poll_job             — check status of a single job
  poll_batch           — check status of a batch job
  list_jobs            — list all jobs
  search               — semantic search across all resumes
  search_skills        — find candidates by skill
  search_experience    — search job history
  list_candidates      — list all indexed candidates
  get_candidate        — get full profile for a specific candidate

Install:
    pip install "mcp[cli]" httpx python-dotenv

Run (Claude Desktop — stdio):
    python mcp_server.py

Claude Desktop config:
    {
      "mcpServers": {
        "resume-pipeline": {
          "command": "/path/to/.venv/bin/python",
          "args": ["/path/to/mcp_server.py"],
          "env": { "API_BASE_URL": "http://localhost:8000" }
        }
      }
    }
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# ── Config ────────────────────────────────────────────────────────────────────

_ROOT = Path(__file__).resolve().parent
load_dotenv(_ROOT / ".env")

API_BASE_URL    = os.getenv("API_BASE_URL",    "http://localhost:8000").rstrip("/")
MCP_SERVER_NAME = os.getenv("MCP_SERVER_NAME", "resume-pipeline")
MCP_LOG_LEVEL   = os.getenv("MCP_LOG_LEVEL",   "INFO").upper()
REQUEST_TIMEOUT = float(os.getenv("MCP_REQUEST_TIMEOUT", "60"))
POLL_INTERVAL   = float(os.getenv("MCP_POLL_INTERVAL",   "2"))
POLL_MAX_WAIT   = float(os.getenv("MCP_POLL_MAX_WAIT",   "300"))

logging.basicConfig(
    level=getattr(logging, MCP_LOG_LEVEL, logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("resume_mcp")

# ── MCP server ────────────────────────────────────────────────────────────────

mcp = FastMCP(MCP_SERVER_NAME)


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url=API_BASE_URL, timeout=REQUEST_TIMEOUT)


def _fmt_error(resp: httpx.Response) -> str:
    try:
        detail = resp.json().get("detail", resp.text)
    except Exception:
        detail = resp.text
    return f"API error {resp.status_code}: {detail}"


async def _poll(url: str) -> dict:
    deadline = time.monotonic() + POLL_MAX_WAIT
    async with _client() as client:
        while True:
            resp = await client.get(url)
            if resp.status_code != 200:
                return {"error": _fmt_error(resp)}
            data = resp.json()
            if data.get("status") in ("done", "failed"):
                return data
            if time.monotonic() > deadline:
                return {**data, "warning": f"Timed out after {POLL_MAX_WAIT}s"}
            await asyncio.sleep(POLL_INTERVAL)


def _mime(path: Path) -> str:
    return {
        ".pdf":  "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".txt":  "text/plain",
    }.get(path.suffix.lower(), "application/octet-stream")


# ── Tools ─────────────────────────────────────────────────────────────────────

@mcp.tool()
async def health_check() -> str:
    """Check if the Resume Pipeline API is running and reachable."""
    try:
        async with _client() as client:
            resp = await client.get("/health")
        if resp.status_code == 200:
            return f"API is healthy. Base URL: {API_BASE_URL}"
        return f"API returned {resp.status_code}: {resp.text}"
    except httpx.ConnectError:
        return f"Cannot reach API at {API_BASE_URL}. Start the server: docker compose up"
    except Exception as exc:
        return f"Health check failed: {exc}"


@mcp.tool()
async def upload_resume(file_path: str) -> str:
    """
    Upload a single resume file for async processing. Returns a job_id.
    This MCP server runs locally and reads the file from your Mac filesystem.

    Args:
        file_path: Full path to a .pdf, .docx, or .txt file on your Mac.
                   Example: /Users/john/Desktop/resume.pdf
    """
    path = Path(file_path)
    if not path.exists():
        return f"File not found: {file_path}"
    if path.suffix.lower() not in {".pdf", ".docx", ".txt"}:
        return f"Unsupported type '{path.suffix}'. Use .pdf, .docx, or .txt"

    async with _client() as client:
        with open(path, "rb") as f:
            resp = await client.post(
                "/resumes/upload",
                files={"file": (path.name, f, _mime(path))},
            )

    if resp.status_code not in (200, 202):
        return _fmt_error(resp)

    data = resp.json()
    return (
        f"Uploaded: {data['file_name']}\n"
        f"job_id: {data['job_id']}\n"
        f"Use poll_job with this job_id to check progress."
    )


@mcp.tool()
async def upload_resume_sync(file_path: str) -> str:
    """
    Upload a single resume and wait for full processing to complete.
    This MCP server runs locally and reads the file from your Mac filesystem.
    Returns candidate name, skills, job history, and chunks stored.

    Args:
        file_path: Full path to a .pdf, .docx, or .txt file on your Mac.
                   Example: /Users/john/Desktop/resume.pdf
    """
    path = Path(file_path)
    if not path.exists():
        return f"File not found: {file_path}"
    if path.suffix.lower() not in {".pdf", ".docx", ".txt"}:
        return f"Unsupported type '{path.suffix}'. Use .pdf, .docx, or .txt"

    async with _client() as client:
        with open(path, "rb") as f:
            resp = await client.post(
                "/resumes/upload/sync",
                files={"file": (path.name, f, _mime(path))},
                timeout=120.0,
            )

    if resp.status_code not in (200, 202):
        return _fmt_error(resp)

    data      = resp.json()
    candidate = data.get("candidate") or {}
    skills    = candidate.get("skills", [])
    jobs      = candidate.get("job_history", [])

    lines = [
        "Resume processed successfully.",
        f"  name     : {candidate.get('name', 'unknown')}",
        f"  email    : {candidate.get('email', 'N/A')}",
        f"  skills   : {', '.join(skills[:10])}{'...' if len(skills) > 10 else ''}",
        f"  roles    : {len(jobs)}",
        f"  chunks   : {data.get('chunks_stored', 0)}",
        f"  duration : {data.get('duration_seconds', 0):.1f}s",
    ]
    if candidate.get("summary"):
        lines.append(f"  summary  : {candidate['summary'][:300]}")
    for j in jobs[:3]:
        lines.append(f"    - {j.get('title')} @ {j.get('company')} ({j.get('start_date')} – {j.get('end_date')})")

    return "\n".join(lines)


@mcp.tool()
async def upload_batch(file_paths: list[str]) -> str:
    """
    Upload multiple resume files in parallel and wait for all to finish.
    Pass a list of file paths or directory paths (directories are expanded automatically).
    This MCP server runs locally and reads files from your Mac filesystem.

    Args:
        file_paths: List of full paths to resume files or directories on your Mac.
                    Example: ["/Users/john/Desktop/candidates/"]
                    Example: ["/Users/john/cv1.pdf", "/Users/john/cv2.docx"]
    """
    resolved: list[Path] = []
    for p in file_paths:
        path = Path(p)
        if path.is_dir():
            for ext in (".pdf", ".docx", ".txt"):
                resolved.extend(path.glob(f"*{ext}"))
                resolved.extend(path.glob(f"*{ext.upper()}"))
        elif path.is_file():
            resolved.append(path)
        else:
            log.warning("Path not found, skipping: %s", p)

    valid = [p for p in resolved if p.suffix.lower() in {".pdf", ".docx", ".txt"}]
    if not valid:
        return f"No supported files found in: {', '.join(file_paths)}"

    log.info("Uploading batch of %d files", len(valid))

    files_payload = [("files", (p.name, open(p, "rb"), _mime(p))) for p in valid]

    try:
        async with _client() as client:
            resp = await client.post("/resumes/batch", files=files_payload)
    finally:
        for _, (_, fh, _) in files_payload:
            fh.close()

    if resp.status_code not in (200, 202):
        return _fmt_error(resp)

    data     = resp.json()
    batch_id = data["batch_id"]

    result = await _poll(f"/batch/{batch_id}")

    lines = [
        f"Batch complete.",
        f"  batch_id  : {batch_id}",
        f"  accepted  : {data['files_accepted']}",
        f"  succeeded : {result.get('succeeded', 0)}",
        f"  failed    : {result.get('failed', 0)}",
    ]
    if result.get("duration_seconds"):
        lines.append(f"  duration  : {result['duration_seconds']:.1f}s")
    if result.get("results"):
        lines.append("\nPer-file results:")
        for r in result["results"]:
            mark = "✓" if r["success"] else "✗"
            lines.append(
                f"  {mark} {r['file']:<30} candidate={r.get('candidate') or 'N/A'}"
            )
            if not r["success"] and r.get("error"):
                lines.append(f"      error: {r['error']}")

    return "\n".join(lines)


@mcp.tool()
async def poll_job(job_id: str) -> str:
    """
    Wait for a single resume upload job to complete and return its result.

    Args:
        job_id: The job_id returned by upload_resume.
    """
    data   = await _poll(f"/jobs/{job_id}")
    status = data.get("status", "unknown")

    if data.get("error"):
        return f"Job {job_id} failed: {data['error']}"

    result = data.get("result") or {}
    return "\n".join([
        f"Job {job_id}: {status}",
        f"  candidate : {result.get('candidate_name', 'N/A')}",
        f"  skills    : {result.get('skills_count', 0)}",
        f"  roles     : {result.get('jobs_count', 0)}",
        f"  chunks    : {result.get('chunks_stored', 0)}",
        f"  duration  : {result.get('duration_seconds', 0):.1f}s",
    ])


@mcp.tool()
async def poll_batch(batch_id: str) -> str:
    """
    Wait for a batch upload job to complete and return the summary.

    Args:
        batch_id: The batch_id returned by upload_batch (when wait_for_completion=False).
    """
    data = await _poll(f"/batch/{batch_id}")

    lines = [
        f"Batch {batch_id}: {data.get('status', 'unknown')}",
        f"  total     : {data.get('total', 0)}",
        f"  succeeded : {data.get('succeeded', 0)}",
        f"  failed    : {data.get('failed', 0)}",
    ]
    if data.get("duration_seconds"):
        lines.append(f"  duration  : {data['duration_seconds']:.1f}s")
    if data.get("results"):
        lines.append("\nPer-file results:")
        for r in data["results"]:
            mark = "✓" if r["success"] else "✗"
            lines.append(f"  {mark} {r['file']:<30} candidate={r.get('candidate') or 'N/A'}")

    return "\n".join(lines)


@mcp.tool()
async def list_jobs() -> str:
    """List all resume processing jobs and their current status."""
    async with _client() as client:
        resp = await client.get("/jobs")

    if resp.status_code != 200:
        return _fmt_error(resp)

    data = resp.json()
    jobs = data.get("jobs", [])

    if not jobs:
        return "No jobs found. Upload a resume to get started."

    lines = [f"Jobs ({data['total']} total):"]
    for j in jobs:
        lines.append(
            f"  {j['status']:<12} {j.get('file_name', 'N/A'):<30} "
            f"id={j['job_id'][:8]}..."
        )
    return "\n".join(lines)


@mcp.tool()
async def search(query: str, top_k: int = 5) -> str:
    """
    Semantic search across all ingested resumes with reranking.

    Args:
        query: Natural language query e.g. "Python engineer with AWS experience".
        top_k: Number of results to return (default 5, max 20).
    """
    async with _client() as client:
        resp = await client.get("/search", params={"q": query, "k": top_k, "rerank": "true"})

    if resp.status_code != 200:
        return _fmt_error(resp)

    data    = resp.json()
    results = data.get("results", [])

    if not results:
        return f"No results found for: '{query}'"

    lines = [f"Search: '{query}' — {data['total']} results"]
    for i, r in enumerate(results, 1):
        score = getattr(r, "get", lambda k, d=None: r.get(k, d))("score")
        lines.append(
            f"\n{i}. [{r.get('candidate', '?')}] section={r.get('section', '?')}\n"
            f"   {r.get('content', '')[:200]}"
        )
    return "\n".join(lines)


@mcp.tool()
async def search_skills(skill: str, top_k: int = 5) -> str:
    """
    Find candidates who have a specific skill.

    Args:
        skill: Skill name e.g. "Python", "Kubernetes", "machine learning".
        top_k: Number of results (default 5, max 20).
    """
    async with _client() as client:
        resp = await client.get("/search/skills", params={"skill": skill, "k": top_k, "rerank": "true"})

    if resp.status_code != 200:
        return _fmt_error(resp)

    data    = resp.json()
    matches = data.get("matches", [])

    if not matches:
        return f"No candidates found with skill: '{skill}'"

    lines = [f"Skill '{skill}' — {data['total']} candidates:"]
    for i, m in enumerate(matches, 1):
        lines.append(f"\n{i}. {m.get('candidate', '?')}\n   {m.get('skills_text', '')[:200]}")
    return "\n".join(lines)


@mcp.tool()
async def search_experience(query: str, top_k: int = 5) -> str:
    """
    Search job history and work experience across all candidates.

    Args:
        query: Experience query e.g. "led a team", "built ML pipeline", "startup founder".
        top_k: Number of results (default 5, max 20).
    """
    async with _client() as client:
        resp = await client.get("/search/experience", params={"q": query, "k": top_k, "rerank": "true"})

    if resp.status_code != 200:
        return _fmt_error(resp)

    data    = resp.json()
    matches = data.get("matches", [])

    if not matches:
        return f"No experience found matching: '{query}'"

    lines = [f"Experience '{query}' — {data['total']} results:"]
    for i, m in enumerate(matches, 1):
        lines.append(
            f"\n{i}. {m.get('candidate', '?')} — {m.get('title', 'N/A')} @ {m.get('company', 'N/A')}\n"
            f"   {m.get('excerpt', '')}"
        )
    return "\n".join(lines)


@mcp.tool()
async def list_candidates() -> str:
    """List all candidates currently indexed in the vector store."""
    async with _client() as client:
        resp = await client.get("/candidates")

    if resp.status_code != 200:
        return _fmt_error(resp)

    data       = resp.json()
    candidates = data.get("candidates", [])

    if not candidates:
        return "No candidates found. Upload some resumes first."

    return (
        f"Indexed candidates ({data['total']} total):\n"
        + "\n".join(f"  {i+1}. {name}" for i, name in enumerate(candidates))
    )


@mcp.tool()
async def get_candidate(name: str) -> str:
    """
    Get the full stored profile for a specific candidate.

    Args:
        name: Candidate name exactly as returned by list_candidates.
    """
    async with _client() as client:
        resp = await client.get(f"/candidates/{name}")

    if resp.status_code == 404:
        return f"Candidate '{name}' not found. Call list_candidates to see available names."
    if resp.status_code != 200:
        return _fmt_error(resp)

    data     = resp.json()
    sections = data.get("sections", {})
    lines    = [f"Profile: {data.get('candidate', name)}", ""]

    for sec in ["summary", "skills", "job_history", "education"] + [
        s for s in sections if s not in ("summary", "skills", "job_history", "education")
    ]:
        chunks = sections.get(sec)
        if not chunks:
            continue
        lines.append(f"── {sec.upper().replace('_', ' ')} ──")
        lines.extend(c.strip() for c in chunks)
        lines.append("")

    return "\n".join(lines)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Resume Pipeline MCP Server")
    parser.add_argument("--transport", choices=["stdio", "sse"], default="stdio")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    log.info("Starting '%s'  transport=%s  api=%s", MCP_SERVER_NAME, args.transport, API_BASE_URL)

    if args.transport == "sse":
        mcp.run(transport="sse", port=args.port)
    else:
        mcp.run(transport="stdio")