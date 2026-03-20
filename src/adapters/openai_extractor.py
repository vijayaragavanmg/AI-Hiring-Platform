"""
Adapter: OpenAILLMExtractor

Implements LLMExtractor using OpenAI via LangChain.
Sends raw resume text to the LLM and parses the JSON response
into a ResumeData entity.
"""

from __future__ import annotations

import json
import logging

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

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


class OpenAILLMExtractor(LLMExtractor):
    """Extract structured ResumeData from raw text using OpenAI."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: int = 0,
    ) -> None:
        """Initialize with model and temperature.
        
        Args:
            model: OpenAI model name (default: gpt-4o-mini)
            temperature: Sampling temperature (default: 0)
        """
        self._chain = _PROMPT | ChatOpenAI(model=model, temperature=temperature)

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
            log.warning("LLM extraction failed: %s", exc)
            raise ExtractionError(str(exc)) from exc
