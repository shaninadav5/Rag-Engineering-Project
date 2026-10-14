"""Stage 11 manual eval against a running server. Writes eval/results.json (gitignored)."""

import json
import time
from pathlib import Path

import httpx

BASE = "http://127.0.0.1:8000"
OUT = Path("eval/results.json")

UPLOADS = [
    ("eval/acme-q3-report.pdf", "Acme Q3 Report"),
    ("eval/acme-handbook.pdf", "Acme Handbook"),
    ("eval/lees-mistake.pdf", "Lee's Mistake"),
]

QUESTIONS = [
    ("q3-revenue", "What was Q3 revenue?", "48 million"),
    ("berlin", "When did the Berlin office open?", "March 2024"),
    ("northstar", "What is the flagship product called?", "Northstar"),
    ("headcount", "What was the employee headcount at quarter end?", "640"),
    ("leave", "How many days of annual leave do full-time staff receive?", "25 days"),
    ("parental", "How long is parental leave?", "16 weeks"),
    ("cafeteria", "When does the office cafeteria close?", "3 pm"),
    ("gettysburg-date", "On what date did Lee order the frontal assault at Gettysburg?", "July 3, 1863"),
    ("day2-casualties", "How many Union casualties had the Confederates inflicted by the end of the second day?", "10,000"),
    ("off-australia", "What is the capital of Australia?", None),
    ("off-pasta", "How do you boil pasta?", None),
    ("near-q1", "What was Q1 revenue?", None),
    ("near-sick", "How many sick days do employees get?", None),
]


def main() -> None:
    client = httpx.Client(base_url=BASE, timeout=180)
    existing = client.get("/documents").json()
    for document in existing:
        client.delete(f"/documents/{document['id']}")

    uploaded = []
    for path, name in UPLOADS:
        started = time.perf_counter()
        with Path(path).open("rb") as handle:
            response = client.post(
                "/documents",
                files={"file": (Path(path).name, handle, "application/pdf")},
                data={"name": name},
            )
        response.raise_for_status()
        document_id = response.json()["id"]
        while True:
            detail = client.get(f"/documents/{document_id}").json()
            if detail["status"] in ("completed", "failed"):
                break
            time.sleep(0.5)
        uploaded.append({**detail, "seconds": round(time.perf_counter() - started, 2)})

    answers = []
    for key, question, expect in QUESTIONS:
        started = time.perf_counter()
        response = client.post("/ask", json={"question": question})
        body = response.json()
        answers.append(
            {
                "key": key,
                "question": question,
                "expect": expect,
                "status": response.status_code,
                "seconds": round(time.perf_counter() - started, 2),
                "answer": body.get("answer"),
                "tokens_used": body.get("tokens_used"),
                "model": body.get("model"),
                "sources": [
                    {
                        "name": source["document_name"],
                        "page": source["page"],
                        "page_end": source["page_end"],
                        "score": source["relevance_score"],
                        "has_expect": bool(expect and expect.lower() in source["chunk_text"].lower()),
                        "text": source["chunk_text"][:240],
                    }
                    for source in body.get("sources", [])
                ],
                "error": body.get("error"),
            }
        )

    OUT.write_text(
        json.dumps({"uploads": uploaded, "answers": answers}, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
