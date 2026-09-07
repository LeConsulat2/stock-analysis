"""Render supplied results; never fetch data or ask a model to format numbers."""

from __future__ import annotations

import html
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from contracts import CompanyReport
from data_tools import finite


def number(value: Any, digits: int = 2) -> str:
    value = finite(value)
    return "unavailable" if value is None else f"{value:,.{digits}f}"


def cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def previous_run(directory: Path, demo: bool) -> dict[str, Any] | None:
    prefix = "DEMO" if demo else "DAILY"
    for path in sorted(directory.glob(f"{prefix}_*.json"), reverse=True):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("schema_version") == 1 and data.get("demo") is demo:
                return data
        except (ValueError, OSError):
            continue
    return None


def write_reports(reports: list[CompanyReport], directory: Path, now: datetime,
                  timezone_name: str, demo: bool, requests: int,
                  prior: dict[str, Any] | None, horizon_years: int = 3) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    local = now.astimezone(ZoneInfo(timezone_name))
    label = "FICTIONAL DEMO - DO NOT INVEST FROM THIS REPORT" if demo else "Daily stock research"
    title = f"{label} | {local.isoformat(timespec='seconds')}"
    lines = [f"# {title}", "", "Research support, not personalized financial advice. No trades were placed.",
             "Prices are the latest completed daily sessions; dates differ by exchange.",
             "A BUY/SELL is a conditional research stance, never a guarantee or a portfolio allocation.",
             f"Research horizon: {horizon_years} years (provisional configurable assumption).",
             f"Hosted API requests: {requests}. Demo uses scripted local sessions.", ""]
    headers = ["Company", "Ticker", "Currency", "Daily close", "Change %", "Session date", "Action"]
    rows = []
    for report in reports:
        data = report.snapshot.data
        rows.append([report.instrument.name, report.instrument.ticker or "not configured",
                     data.get("currency") or "-", number(data.get("close")),
                     number(data.get("change_pct")), data.get("session_date") or "unavailable",
                     ("DEMO " if demo else "") + report.published_action])
    lines += ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in rows]
    lines += ["", "Change % uses adjusted closes across consecutive observed sessions; displayed prices are unadjusted.",
              "No price is invented for an organization without a configured direct listing.", ""]
    old = {c["instrument"]["key"]: c for c in (prior or {}).get("companies", [])}
    for report in reports:
        item = report.instrument
        lines += [f"## {item.name}", "", f"Published stance: {report.published_action}", item.note]
        if report.blocks:
            lines += ["", "Publication gate:"] + [f"- {b}" for b in report.blocks]
        previous = old.get(item.key)
        if previous:
            prior_urls = {a["url"] for e in previous.get("evidence", [])
                          for a in e["data"].get("articles", [])}
            current_urls = {a["url"] for e in report.evidence for a in e.data.get("articles", [])}
            lines += ["", f"Previous report stance: {previous['published_action']}. "
                      f"New covered news URLs since that report: {len(current_urls - prior_urls)}."]
        else:
            lines += ["", "First retained report for this entry; no prior comparison available."]
        for agent in report.analysts + [report.decision, report.skeptic]:
            lines += ["", f"### {agent.role} ({agent.provider} / {agent.model})", f"Status: {agent.status}"]
            if agent.error:
                lines.append(agent.error)
            if agent.content:
                lines += [agent.content["summary"], "", "Rationale: " + agent.content["rationale"],
                          "Change my view if: " + agent.content["invalidation"]]
                for claim in agent.content["claims"]:
                    lines.append(f"- {claim['text']} [{', '.join(claim['evidence_ids'])}]")
                lines += [f"- Risk: {r}" for r in agent.content["risks"]]
                if agent.role == "decision":
                    lines.append("Draft model stance (subject to gate): " + agent.content["stance"])
        lines += ["", "### Evidence and news", ""]
        for evidence in report.evidence:
            lines += [f"Evidence {evidence.id}: {evidence.source}; as of {evidence.as_of or 'see item timestamps'}."]
            if evidence.error:
                lines.append("ERROR: " + evidence.error)
            lines += ["Warning: " + w for w in evidence.warnings]
            for article in evidence.data.get("articles", []):
                lines += [f"- {article['published_at']} | {article['title']}",
                          f"  {article['url']}", f"  {article.get('summary', '')}"]
            if evidence.kind == "valuation":
                lines += [f"Conditional present value: {number(evidence.data['present_value'])} "
                          f"{evidence.data['currency']}; inputs: {json.dumps(evidence.data['inputs'])}"]
    lines += ["", "## Practical limits", "",
              "Yahoo coverage can omit relevant news or contain stale/mismatched data. "
              "Headlines and summaries do not establish the full story. Relationship leads are not verified contracts.",
              "The gate checks structure, provenance IDs, freshness and valuation consistency. "
              "It cannot prove every sentence is true; a human must inspect important original sources.",
              "No portfolio, tax, suitability, broker execution, email delivery, or full-text web crawler is included."]
    markdown = "\n".join(lines) + "\n"
    table = "<table><thead><tr>" + "".join(f"<th>{html.escape(h)}</th>" for h in headers) + "</tr></thead><tbody>"
    for row in rows:
        table += "<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in row) + "</tr>"
    table += "</tbody></table>"
    document = ("<!doctype html><html lang='en'><meta charset='utf-8'>"
                "<meta name='viewport' content='width=device-width, initial-scale=1'>"
                f"<title>{html.escape(label)}</title><style>"
                "body{font:16px system-ui;margin:32px auto;max-width:1150px;padding:0 20px;color:#172b3a}"
                "table{border-collapse:collapse;width:100%;font-size:14px}"
                "td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left}"
                "th{background:#eaf2f5}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:15px/1.6 system-ui}"
                "</style><body>"
                f"<h1>{html.escape(label)}</h1><p>{html.escape(local.isoformat())}</p>"
                + table + "<h2>Analysis and evidence</h2><pre>" + html.escape(markdown) + "</pre></body></html>")
    stem = ("DEMO" if demo else "DAILY") + "_" + now.strftime("%Y%m%d_%H%M%S_%fZ")
    audit = {"schema_version": 1, "demo": demo, "generated_at": now.isoformat(),
             "display_timezone": timezone_name, "horizon_years": horizon_years, "requests": requests,
             "companies": [asdict(r) for r in reports]}
    paths = []
    for extension, content in (("md", markdown), ("html", document),
                               ("json", json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False))):
        path = directory / f"{stem}.{extension}"
        # Unique, exclusive filenames preserve every prior decision record.
        with path.open("x", encoding="utf-8") as stream:
            stream.write(content)
        paths.append(path)
    return paths
