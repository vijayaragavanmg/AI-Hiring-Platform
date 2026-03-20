"""
Adapter: GeminiLLMExtractor

Implements LLMExtractor using Google Gemini via LangChain.
Drop-in replacement for OpenAILLMExtractor — same port, different model.

Install:
    pip install langchain-google-genai

.env:
    GEMINI_API_KEY=your-key
    GEMINI_LLM_MODEL=gemini-1.5-flash       # or gemini-1.5-pro
    GEMINI_LLM_TEMPERATURE=0
"""

from __future__ import annotations

import json
import logging

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from src.domain.config import GEMINI_API_KEY, GEMINI_LLM_MODEL, GEMINI_LLM_TEMPERATURE
from src.domain.entities import ResumeData
from src.ports.llm_extractor import ExtractionError, LLMExtractor

log = logging.getLogger(__name__)

_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert resume parser.
Extract the following from the resume text and return ONLY valid JSON — no markdown, no preamble.

Schema:
{{
  "name": "string",
  "email": "string or null",
  "phone": "string or null",
  "summary": "2-4 sentence professional summary",
  "skills": ["skill1", "skill2", ...],
  "job_history": [
    {{
      "company": "string",
      "title": "string",
      "start_date": "string",
      "end_date": "string",
      "responsibilities": ["string", ...]
    }}
  ],
  "education": ["Degree, Institution, Year", ...]
}}

Rules:
- skills: flat list of individual skills (languages, tools, frameworks, soft skills)
- job_history: reverse chronological order (most recent first)
- summary: synthesise one from the content if not explicitly present
- Return only JSON — no explanation."""),
    ("human", "Resume text:\n\n{resume_text}"),
])


class GeminiLLMExtractor(LLMExtractor):

    def __init__(
        self,
        model:       str   = GEMINI_LLM_MODEL,
        temperature: float = GEMINI_LLM_TEMPERATURE,
        api_key:     str   = GEMINI_API_KEY,
    ) -> None:
        llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            google_api_key=api_key,
            convert_system_message_to_human=True,  # Gemini requires this
        )
        self._chain = _PROMPT | llm

    def extract(self, raw_text: str) -> ResumeData:
        try:
            response = self._chain.invoke({"resume_text": raw_text})
            raw_json = response.content.strip()

            # Strip accidental markdown fences
            if raw_json.startswith("```"):
                raw_json = raw_json.split("```")[1]
                if raw_json.startswith("json"):
                    raw_json = raw_json[4:]

            return ResumeData(**json.loads(raw_json))

        except Exception as exc:
            log.warning("Gemini extraction failed: %s", exc)
            raise ExtractionError(str(exc)) from exc
