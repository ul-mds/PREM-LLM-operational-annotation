# PREM-LLM-Operational -Annotation

This repository contains the code, annotation framework, and analysis workflow for a medical informatics study investigating how **Large Language Models (LLMs)** can transform free-text **Patient Reported Experience Measures (PREMs)** into **structured, operational -ready signals** for use in health information systems.

The project focuses on converting unstructured patient feedback into a structured schema including:

- primary issue category
- severity
- urgency
- evidence span
- suggested operational action

The overall aim is to evaluate whether LLMs can support the operational integration of patient experience narratives and operational healthcare quality management workflows.

---

## Project Scope

Free-text patient comments often contain valuable information about:

- access barriers
- communication problems
- staff attitude
- administrative issues
- continuity of care
- perceived clinical quality or safety

However, these data are difficult to process systematically in routine healthcare settings. This repository supports a structured pipeline for:

1. manual annotation of PREM comments
2. AI-based annotation using LLMs
3. comparison of human and AI annotations
4. assessment of structural output quality
5. preparation of operational -oriented outputs in JSON format

---

## Annotation Schema

Each comment is annotated using the following fields:

- `label_primary`
- `severity`
- `urgency`
- `evidence_span`
- `suggested_operational_action`
- `notes`

### Primary label taxonomy

- `Access/Delay`
- `Communication`
- `Clinical Quality/Safety`
- `Staff Attitude/Respect`
- `Administrative/Process`
- `Continuity/Coordination`
- `Positive Feedback`
- `Other/Unclear`

---

## Repository Structure

```text
PREM-LLM-operational-Annotation/
│
├── data/
│   ├── raw/                    # Raw input files
│   ├── annotation/             # Human annotation files
│   └── processed/              # AI-annotated and merged datasets
│
├── scripts/
│   ├── annotate_claude.py      # Claude-based annotation pipeline
│   ├── annotate_openai.py      # OpenAI-based annotation pipeline
│   ├── validate_outputs.py     # JSON/schema/evidence validation
│   └── analysis.R              # Agreement and performance analysis
│
├── docs/
│   ├── annotation_guideline.xlsx
│   └── study_protocol.md
│
├── results/
│   ├── tables/
│   └── figures/
│
└── README.md
```

---

## Input Data Format

The expected input file is an Excel file containing at least the following column:

- `prem_text`

Optional metadata columns may include:

- `row_id`
- `organisation_name`
- `response_date`

Example input structure:

| row_id | organisation_name | response_date | prem_text |
|-------|-------------------|---------------|-----------|
| 1 | Example GP Practice | 2010-06-01 | TITLE: ... LIKED: ... DISLIKED: ... ADVICE: ... |

---

## Output Data Format

The annotation scripts generate an output Excel file with the original data plus the following AI-generated columns:

- `ai_label_primary`
- `ai_severity`
- `ai_urgency`
- `ai_evidence_span`
- `ai_suggested_operational_action`
- `ai_notes`

Additional validation columns:

- `ai_json_valid`
- `ai_schema_ok`
- `ai_values_ok`
- `ai_evidence_in_text`
- `ai_evidence_too_long`
- `ai_error`

These fields are used to assess the structural reliability of the LLM outputs.

---

## Annotation Workflow

### Human annotation
A subset of comments is manually annotated by independent annotators using a predefined annotation guideline.

### AI annotation
LLMs are prompted to return structured outputs in JSON format.

### Validation
Outputs are automatically checked for:

- JSON validity
- schema compliance
- valid taxonomy values
- evidence span grounding in source text
- evidence length

### Comparison
Human and AI annotations can then be compared using:

- accuracy
- macro F1
- Cohen’s kappa
- class-level error analysis

---

## Installation

install dependencies:

```bash
pip install pandas openpyxl anthropic openai
```

Depending on the pipeline you use, you may need either:

- `ANTHROPIC_API_KEY`
- `OPENAI_API_KEY`

---

## Running the Annotation Pipeline

### Claude version

```python
import os
import json
import time
import re
import pandas as pd
from typing import Dict, Any, Optional
import anthropic

# =========================
# Config
# =========================

INPUT_XLSX = "annotation_sample_500.xlsx"
INPUT_PATH = f"/home/mds/Desktop/R_Folder/{INPUT_XLSX}"
OUTPUT_PATH = "/home/mds/Desktop/annotated_sample_500_claude.xlsx"

MODEL = "claude-sonnet-4-6"
SLEEP_SEC = 0.25
MAX_RETRIES = 3
MAX_TOKENS = 1024

ALLOWED_LABELS = {
    "Access/Delay",
    "Communication",
    "Clinical Quality/Safety",
    "Staff Attitude/Respect",
    "Administrative/Process",
    "Continuity/Coordination",
    "Positive Feedback",
    "Other/Unclear",
}
ALLOWED_SEVERITY = {"low", "moderate", "high"}
ALLOWED_URGENCY = {"routine", "soon", "urgent"}

SYSTEM_PROMPT = """You are an AI annotator for a medical informatics study. Annotate one PREM text for operational-oriented structuring.

You MUST output ONLY valid JSON with the exact keys: label_primary, severity, urgency, evidence_span, suggested_operational_action, notes

Rules:
- label_primary: choose EXACTLY ONE from: "Access/Delay", "Communication", "Clinical Quality/Safety", "Staff Attitude/Respect", "Administrative/Process", "Continuity/Coordination", "Positive Feedback", "Other/Unclear"
- severity: low|moderate|high
- urgency: routine|soon|urgent
- evidence_span: copy 1–2 sentences VERBATIM from the input text. Do NOT paraphrase, do NOT summarize, do NOT add commentary. Do NOT copy a whole paragraph. Max ~300 characters.
- suggested_operational_action: one short operational/operational action (not medical treatment advice).
- notes: optional, brief.

Decision rules:
- Safety/harm/medication error/misdiagnosis/negligence -> Clinical Quality/Safety
- Referral delay purely procedural -> Administrative/Process
- Referral delay + clinical management criticism -> Clinical Quality/Safety
- Lack of explanation/listening -> Communication
- Rude/disrespectful -> Staff Attitude/Respect
- If multiple issues, choose most critical/actionable; if equal prefer:
  Clinical Quality/Safety > Access/Delay > Administrative/Process > Communication > Staff Attitude/Respect > Continuity/Coordination > Positive Feedback > Other/Unclear
"""

def extract_json(text: str) -> Dict[str, Any]:
    """Extract JSON object from model response."""
    text = text.strip()
    text = re.sub(r"^(?:json)?\s*", "", text)
    text = re.sub(r"\s*$", "", text)
    return json.loads(text)

def validate_output(obj: Dict[str, Any], prem_text: str) -> Dict[str, Any]:
    """Validate schema + allowed values + evidence span present in prem_text."""
    flags = {
        "json_valid": True,
        "schema_ok": True,
        "values_ok": True,
        "evidence_in_text": True,
        "evidence_too_long": False,
    }

    required = [
        "label_primary",
        "severity",
        "urgency",
        "evidence_span",
        "suggested_operational_action",
        "notes",
    ]

    for k in required:
        if k not in obj:
            flags["schema_ok"] = False

    label = obj.get("label_primary", "")
    sev = obj.get("severity", "")
    urg = obj.get("urgency", "")
    ev = obj.get("evidence_span", "")

    if label not in ALLOWED_LABELS or sev not in ALLOWED_SEVERITY or urg not in ALLOWED_URGENCY:
        flags["values_ok"] = False

    if isinstance(ev, str):
        if len(ev) > 350:
            flags["evidence_too_long"] = True

        norm_ev = " ".join(ev.split())
        norm_text = " ".join(str(prem_text).split())

        if norm_ev and norm_ev not in norm_text:
            flags["evidence_in_text"] = False
    else:
        flags["evidence_in_text"] = False

    return flags

def annotate_one(client: anthropic.Anthropic, prem_text: str) -> Dict[str, Any]:
    user_prompt = f"""Annotate this PREM text.

Return ONLY valid JSON. Do not add any explanation before or after the JSON.

<<<PREM_TEXT_START
{prem_text}
PREM_TEXT_END>>>
"""

    last_err: Optional[Exception] = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )

            text = response.content[0].text
            obj = extract_json(text)
            return obj

        except Exception as e:
            last_err = e
            time.sleep(1.0 * attempt)

    raise RuntimeError(f"Failed after {MAX_RETRIES} retries: {last_err}")

def main():
    start_time = time.time()

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("Please set ANTHROPIC_API_KEY as an environment variable.")

    client = anthropic.Anthropic(api_key=api_key)
    df = pd.read_excel(INPUT_PATH)

    if "prem_text" not in df.columns:
        raise RuntimeError("Input file must contain a 'prem_text' column.")

    outputs = []
    total_rows = len(df)

    for i, row in df.iterrows():
        print(f"Processing row {i + 1} / {total_rows}")
        prem_text = row["prem_text"]

        if not isinstance(prem_text, str) or not prem_text.strip():
            outputs.append({
                "ai_label_primary": "",
                "ai_severity": "",
                "ai_urgency": "",
                "ai_evidence_span": "",
                "ai_suggested_operational_action": "",
                "ai_notes": "",
                "ai_json_valid": False,
                "ai_schema_ok": False,
                "ai_values_ok": False,
                "ai_evidence_in_text": False,
                "ai_evidence_too_long": False,
                "ai_error": "Empty prem_text",
            })
            continue

        try:
            obj = annotate_one(client, prem_text)
            flags = validate_output(obj, prem_text)

            outputs.append({
                "ai_label_primary": obj.get("label_primary", ""),
                "ai_severity": obj.get("severity", ""),
                "ai_urgency": obj.get("urgency", ""),
                "ai_evidence_span": obj.get("evidence_span", ""),
                "ai_suggested_operational_action": obj.get("suggested_operational_action", ""),
                "ai_notes": obj.get("notes", ""),
                **{f"ai_{k}": v for k, v in flags.items()},
                "ai_error": "",
            })

        except Exception as e:
            outputs.append({
                "ai_label_primary": "",
                "ai_severity": "",
                "ai_urgency": "",
                "ai_evidence_span": "",
                "ai_suggested_operational_action": "",
                "ai_notes": "",
                "ai_json_valid": False,
                "ai_schema_ok": False,
                "ai_values_ok": False,
                "ai_evidence_in_text": False,
                "ai_evidence_too_long": False,
                "ai_error": str(e),
            })

        time.sleep(SLEEP_SEC)

    out_df = pd.concat([df.reset_index(drop=True), pd.DataFrame(outputs)], axis=1)
    out_df.to_excel(OUTPUT_PATH, index=False)
    print("Saved:", OUTPUT_PATH)

    elapsed = time.time() - start_time
    print(f"Total processing time: {elapsed:.2f} seconds")
    print(f"Total processing time: {elapsed/60:.2f} minutes")

if __name__ == "__main__":
    main()
```

### OpenAI version


```python
import json
import time
import re
import pandas as pd
from typing import Dict, Any, Optional
from openai import OpenAI

# =========================
# Config
# =========================
INPUT_XLSX = "annotation_sample_500.xlsx"
INPUT_PATH = f"/home/mds/Desktop/R_Folder/{INPUT_XLSX}"   
OUTPUT_PATH = "/home/mds/Desktop/annotated_sample_500.xlsx"

MODEL = "gpt-5.2"
SLEEP_SEC = 0.25
MAX_RETRIES = 3

ALLOWED_LABELS = {
    "Access/Delay",
    "Communication",
    "Clinical Quality/Safety",
    "Staff Attitude/Respect",
    "Administrative/Process",
    "Continuity/Coordination",
    "Positive Feedback",
    "Other/Unclear",
}
ALLOWED_SEVERITY = {"low", "moderate", "high"}
ALLOWED_URGENCY = {"routine", "soon", "urgent"}

SYSTEM_PROMPT = """You are an AI annotator for a medical informatics study.
Annotate one PREM text for operational-oriented structuring.

You MUST output ONLY valid JSON with the exact keys:
label_primary, severity, urgency, evidence_span, suggested_operational_action, notes

Rules:
- label_primary: choose EXACTLY ONE from:
  "Access/Delay", "Communication", "Clinical Quality/Safety", "Staff Attitude/Respect",
  "Administrative/Process", "Continuity/Coordination", "Positive Feedback", "Other/Unclear"
- severity: low|moderate|high
- urgency: routine|soon|urgent
- evidence_span: copy 1–2 sentences VERBATIM from the input text.
  Do NOT paraphrase, do NOT summarize, do NOT add commentary.
  Do NOT copy a whole paragraph. Max ~300 characters.
- suggested_operational_action: one short operational/operational action (not medical treatment advice).
- notes: optional, brief.

Decision rules:
- Safety/harm/medication error/misdiagnosis/negligence -> Clinical Quality/Safety
- Referral delay purely procedural -> Administrative/Process
- Referral delay + clinical management criticism -> Clinical Quality/Safety
- Lack of explanation/listening -> Communication
- Rude/disrespectful -> Staff Attitude/Respect
- If multiple issues, choose most critical/actionable; if equal prefer:
  Clinical Quality/Safety > Access/Delay > Administrative/Process > Communication >
  Staff Attitude/Respect > Continuity/Coordination > Positive Feedback > Other/Unclear
"""

def extract_json(text: str) -> Dict[str, Any]:
    """Extract JSON object from model response."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text)

def validate_output(obj: Dict[str, Any], prem_text: str) -> Dict[str, Any]:
    """Validate schema + allowed values + evidence span present in prem_text."""
    flags = {
        "json_valid": True,
        "schema_ok": True,
        "values_ok": True,
        "evidence_in_text": True,
        "evidence_too_long": False,
    }

    required = [
        "label_primary",
        "severity",
        "urgency",
        "evidence_span",
        "suggested_operational_action",
        "notes",
    ]

    for k in required:
        if k not in obj:
            flags["schema_ok"] = False

    label = obj.get("label_primary", "")
    sev = obj.get("severity", "")
    urg = obj.get("urgency", "")
    ev = obj.get("evidence_span", "")

    if label not in ALLOWED_LABELS or sev not in ALLOWED_SEVERITY or urg not in ALLOWED_URGENCY:
        flags["values_ok"] = False

    if isinstance(ev, str):
        if len(ev) > 350:
            flags["evidence_too_long"] = True

        norm_ev = " ".join(ev.split())

        norm_text = " ".join(str(prem_text).split())

        if norm_ev and norm_ev not in norm_text:
            flags["evidence_in_text"] = False
    else:
        flags["evidence_in_text"] = False

    return flags

def annotate_one(client: OpenAI, prem_text: str) -> Dict[str, Any]:
    user_prompt = f"""Annotate this PREM text:

<<<PREM_TEXT_START
{prem_text}
PREM_TEXT_END>>>
"""

    last_err: Optional[Exception] = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.responses.create(
                model=MODEL,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
            text = resp.output_text
            obj = extract_json(text)
            return obj

        except Exception as e:
            last_err = e
            time.sleep(1.0 * attempt)

    raise RuntimeError(f"Failed after {MAX_RETRIES} retries: {last_err}")

def main():
    start_time = time.time()
    api_key = "***"
    if not api_key:
        raise RuntimeError("API key is missing.")


    client = OpenAI(api_key=api_key)

    df = pd.read_excel(INPUT_PATH)

    # TEST için sadece ilk 10 satırı çalıştır
    #df = df.head(10)
    if "prem_text" not in df.columns:
        raise RuntimeError("Input file must contain a 'prem_text' column.")

    outputs = []

    for _, row in df.iterrows():
        prem_text = row["prem_text"]

        if not isinstance(prem_text, str) or not prem_text.strip():
            outputs.append({
                "ai_label_primary": "",
                "ai_severity": "",
                "ai_urgency": "",
                "ai_evidence_span": "",
                "ai_suggested_operational_action": "",
                "ai_notes": "",
                "ai_json_valid": False,
                "ai_schema_ok": False,
                "ai_values_ok": False,
                "ai_evidence_in_text": False,
                "ai_evidence_too_long": False,
                "ai_error": "Empty prem_text",
            })
            continue

        try:
            obj = annotate_one(client, prem_text)
            flags = validate_output(obj, prem_text)

            outputs.append({
                "ai_label_primary": obj.get("label_primary", ""),
                "ai_severity": obj.get("severity", ""),
                "ai_urgency": obj.get("urgency", ""),
                "ai_evidence_span": obj.get("evidence_span", ""),
                "ai_suggested_operational_action": obj.get("suggested_operational_action", ""),
                "ai_notes": obj.get("notes", ""),
                **{f"ai_{k}": v for k, v in flags.items()},
                "ai_error": "",
            })

        except Exception as e:
            outputs.append({
                "ai_label_primary": "",
                "ai_severity": "",
                "ai_urgency": "",
                "ai_evidence_span": "",
                "ai_suggested_operational_action": "",
                "ai_notes": "",
                "ai_json_valid": False,
                "ai_schema_ok": False,
                "ai_values_ok": False,
                "ai_evidence_in_text": False,
                "ai_evidence_too_long": False,
                "ai_error": str(e),
            })

        time.sleep(SLEEP_SEC)

    out_df = pd.concat([df.reset_index(drop=True), pd.DataFrame(outputs)], axis=1)
    out_df.to_excel(OUTPUT_PATH, index=False)
    print("Saved:", OUTPUT_PATH)
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Total processing time: {elapsed:.2f} seconds")
    print(f"Total processing time: {elapsed/60:.2f} minutes")
if __name__ == "__main__":
    main()
```
---

## Methodological Notes

This repository is designed for a study in which PREM comments are converted into structured, operationally meaningful signals for healthcare information systems.

The project is not limited to text classification alone. It also evaluates whether LLM outputs are:

- structurally valid
- evidence-grounded
- compatible with operational -oriented downstream use

This makes the pipeline relevant not only for NLP benchmarking but also for **health information systems research** and **AI-enabled quality improvement workflows**.

---

## Important Caveats

- Free-text evidence matching may underestimate grounding when the source text contains formatting differences such as HTML entities (e.g. `&quot;` vs `"`).
- Human–AI agreement should only be interpreted as valid when human annotations are generated independently from the model outputs.
- Suggested operational  actions are operational recommendations and are **not** treatment recommendations.

---

## Planned Analyses

The repository supports the following analyses:

- inter-rater agreement between human annotators
- human vs AI agreement
- JSON validity and schema compliance
- evidence grounding
- error analysis across similar categories
- operational  mapping of structured PREM outputs

---

## Citation

Data source: NHS England. NHS oversight framework csv metadata file – Q2 2025/26 (Publication reference PRN01916, published September 9, 2025; updated December 15, 2025). Available from: NHS England. https://www.england.nhs.uk/long-read/nhs-oversight-framework-csv-metadata-file/


## Contact

For questions related to the study design, annotation framework, or analysis pipeline, please contact the repository owner.
