
import os
import sys
import json
import subprocess
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import OpenAI

app = FastAPI(title="Code Interpreter with AI Error Analysis")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CodeRequest(BaseModel):
    code: str


class ErrorAnalysis(BaseModel):
    error_lines: List[int] = Field(default_factory=list)


def execute_python_code(code: str) -> dict:
    """Execute Python code and capture its output or traceback."""
    try:
        process = subprocess.run(
            [sys.executable, "-I", "-c", code],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if process.returncode == 0:
            return {
                "success": True,
                "output": process.stdout,
            }

        return {
            "success": False,
            "output": process.stdout + process.stderr,
        }

    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")

        stderr = exc.stderr or ""
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")

        return {
            "success": False,
            "output": output + stderr + "\nExecution timed out after 5 seconds.",
        }


def analyze_error_with_ai(code: str, traceback_text: str) -> List[int]:
    """Use AI Pipe to identify the failing source-code line numbers."""
    token = os.environ.get("AIPIPE_TOKEN") or os.environ.get(
        "OPENAI_API_KEY"
    )

    if not token:
        raise RuntimeError("AIPIPE_TOKEN environment variable is missing.")

    client = OpenAI(
        api_key=token,
        base_url="https://aipipe.org/openai/v1",
    )

    prompt = f"""
Identify the 1-based line numbers in the submitted Python code
that caused the execution error.

Return a JSON object with exactly this structure:
{{"error_lines": [3]}}

Return an empty list only if no source-code line can be identified.
Do not report line numbers belonging to the Python interpreter itself.
Use the traceback as evidence and do not invent line numbers.

SUBMITTED CODE:
{code}

EXECUTION OUTPUT:
{traceback_text}

There are {len(code.splitlines())} lines in the submitted code.
"""

    response = client.chat.completions.create(
        model="openai/gpt-4.1-nano",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise Python traceback analyst. "
                    "Return only valid JSON with an error_lines array."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )

    content = response.choices[0].message.content or "{}"
    analysis = ErrorAnalysis.model_validate_json(content)

    # Keep only valid, 1-based lines in the submitted code.
    line_count = len(code.splitlines())
    return sorted(
        {
            line
            for line in analysis.error_lines
            if isinstance(line, int) and 1 <= line <= line_count
        }
    )


@app.post("/code-interpreter")
def code_interpreter(request: CodeRequest):
    execution = execute_python_code(request.code)

    if execution["success"]:
        return {
            "error": [],
            "result": execution["output"],
        }

    error_lines = analyze_error_with_ai(
        request.code,
        execution["output"],
    )

    return {
        "error": error_lines,
        "result": execution["output"],
    }
