#!/usr/bin/env python3
"""
Local Reply-Quality Human Annotation Web App

Runs a lightweight, zero-dependency local web interface using Python's standard http.server.
Allows annotators to evaluate 50 heldout examples one-by-one on 6 dimensions (1-5 scale)
plus optional comments.

Usage:
    python3 evaluation/reply_quality/annotate_replies.py [--port 8500]

Data Persistence:
    Saves to: evaluation/reply_quality/human_annotations.csv
    Resume: Automatically loads existing annotations if present.
"""

import os
import sys
import csv
import json
import argparse
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_PATH = os.path.join(BASE_DIR, 'human_annotation_template.csv')
ANNOTATIONS_PATH = os.path.join(BASE_DIR, 'human_annotations.csv')

RATING_DEFINITIONS = {
    "relevance": {
        "title": "1. RELEVANCE",
        "desc": "Does the reply address the customer's actual support issue?",
        "scale": "1 = Irrelevant/Off-topic | 3 = Partially addresses issue | 5 = Directly & fully addresses issue"
    },
    "groundedness": {
        "title": "2. GROUNDEDNESS",
        "desc": "Is the reply consistent with retrieved historical AmazonHelp evidence & avoids unsupported claims?",
        "scale": "1 = Hallucinates/Contradicts evidence | 3 = Mostly consistent | 5 = Fully grounded & factual"
    },
    "helpfulness": {
        "title": "3. HELPFULNESS",
        "desc": "Does it provide an actionable & appropriate next step or routing?",
        "scale": "1 = Unhelpful/Dead-end | 3 = Moderate help/generic | 5 = Actionable, clear & accurate next steps"
    },
    "tone": {
        "title": "4. TONE",
        "desc": "Is it professional, concise, empathetic, and suitable for customer support?",
        "scale": "1 = Rude/Inappropriate/Overly long | 3 = Acceptable tone | 5 = Empathetic, crisp & highly professional"
    },
    "safety": {
        "title": "5. SAFETY / ESCALATION CONSISTENCY",
        "desc": "Does it avoid claiming private account/order actions publicly? Directs to private channel if needed?",
        "scale": "1 = Unsafe/Claims private actions publicly | 3 = Acceptable safety | 5 = Safe routing, proper DM guidance"
    },
    "overall": {
        "title": "6. OVERALL QUALITY",
        "desc": "Overall, would this be a reasonable first-draft response for a human support agent?",
        "scale": "1 = Poor (Do not send) | 3 = Acceptable draft (Needs edit) | 5 = Excellent (Ready to send)"
    }
}

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Reply-Quality Human Annotation Tool</title>
    <style>
        :root {
            --bg-color: #0f172a;
            --card-bg: #1e293b;
            --border-color: #334155;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --accent-blue: #38bdf8;
            --accent-green: #22c55e;
            --accent-amber: #f59e0b;
            --accent-purple: #a855f7;
            --danger-red: #ef4444;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-main);
            line-height: 1.5;
            padding: 20px;
        }

        .container {
            max-width: 1100px;
            margin: 0 auto;
        }

        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: var(--card-bg);
            padding: 16px 24px;
            border-radius: 12px;
            border: 1px solid var(--border-color);
            margin-bottom: 20px;
        }

        h1 { font-size: 1.25rem; font-weight: 600; color: var(--accent-blue); }

        .progress-bar-container {
            width: 300px;
            background: #0f172a;
            height: 12px;
            border-radius: 6px;
            overflow: hidden;
            border: 1px solid var(--border-color);
        }

        .progress-bar-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--accent-blue), var(--accent-green));
            width: 0%;
            transition: width 0.3s ease;
        }

        .stats-badge {
            font-size: 0.85rem;
            color: var(--text-muted);
            font-weight: 500;
        }

        .card {
            background: var(--card-bg);
            border-radius: 12px;
            border: 1px solid var(--border-color);
            padding: 20px;
            margin-bottom: 20px;
        }

        .card-header {
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--accent-blue);
            font-weight: 700;
            margin-bottom: 10px;
            display: flex;
            justify-content: space-between;
        }

        .badge {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 600;
        }
        .badge-true { background: rgba(239, 68, 68, 0.2); color: var(--danger-red); border: 1px solid var(--danger-red); }
        .badge-false { background: rgba(34, 197, 94, 0.2); color: var(--accent-green); border: 1px solid var(--accent-green); }
        .badge-intent { background: rgba(56, 189, 248, 0.2); color: var(--accent-blue); border: 1px solid var(--accent-blue); }

        .box {
            background: #0f172a;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 12px 16px;
            font-size: 0.95rem;
            white-space: pre-wrap;
            word-break: break-word;
            margin-bottom: 12px;
        }

        .grid-2 {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
        }

        .rating-section {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
        }

        .rating-box {
            background: #0f172a;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 14px;
        }

        .rating-title {
            font-size: 0.9rem;
            font-weight: 700;
            color: var(--text-main);
            margin-bottom: 4px;
        }

        .rating-desc {
            font-size: 0.8rem;
            color: var(--text-muted);
            margin-bottom: 10px;
        }

        .scale-guide {
            font-size: 0.75rem;
            color: var(--accent-amber);
            margin-bottom: 10px;
        }

        .btn-group {
            display: flex;
            gap: 8px;
        }

        .rating-btn {
            flex: 1;
            padding: 8px;
            background: #1e293b;
            border: 1px solid var(--border-color);
            color: var(--text-main);
            border-radius: 6px;
            cursor: pointer;
            font-weight: 600;
            font-size: 0.9rem;
            transition: all 0.15s ease;
        }

        .rating-btn:hover {
            border-color: var(--accent-blue);
            background: #334155;
        }

        .rating-btn.active {
            background: var(--accent-blue);
            color: #0f172a;
            border-color: var(--accent-blue);
            font-weight: 700;
        }

        .comment-box {
            width: 100%;
            background: #0f172a;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 12px;
            color: var(--text-main);
            font-family: inherit;
            resize: vertical;
            min-height: 70px;
            margin-top: 8px;
        }

        .nav-controls {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 20px;
        }

        .nav-btn {
            padding: 10px 24px;
            border-radius: 8px;
            font-weight: 600;
            font-size: 0.95rem;
            cursor: pointer;
            border: 1px solid var(--border-color);
            background: var(--card-bg);
            color: var(--text-main);
            transition: all 0.2s ease;
        }

        .nav-btn:hover:not(:disabled) {
            background: #334155;
            border-color: var(--accent-blue);
        }

        .nav-btn:disabled {
            opacity: 0.4;
            cursor: not-allowed;
        }

        .nav-btn-primary {
            background: var(--accent-green);
            color: #0f172a;
            border: none;
        }

        .nav-btn-primary:hover:not(:disabled) {
            background: #16a34a;
        }

        .status-toast {
            font-size: 0.85rem;
            color: var(--accent-green);
            font-weight: 600;
            opacity: 0;
            transition: opacity 0.3s ease;
        }

        .status-toast.show {
            opacity: 1;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div>
                <h1>Human Reply-Quality Annotation</h1>
                <div class="stats-badge" id="nav-info">Example 1 of 50</div>
            </div>
            <div style="text-align: right;">
                <div class="stats-badge" id="completion-count" style="margin-bottom: 6px;">0 / 50 Completed</div>
                <div class="progress-bar-container">
                    <div class="progress-bar-fill" id="progress-fill"></div>
                </div>
            </div>
        </header>

        <!-- Context & Details Card -->
        <div class="card">
            <div class="card-header">
                <span>Current Conversation & Metadata</span>
                <div>
                    <span class="badge badge-intent" id="badge-intent">Intent</span>
                    <span class="badge" id="badge-escalation">Escalation</span>
                </div>
            </div>
            
            <div class="grid-2">
                <div>
                    <div style="font-size: 0.8rem; color: var(--text-muted); margin-bottom: 4px; font-weight: 600;">CUSTOMER MESSAGE</div>
                    <div class="box" id="customer-message" style="color: #7dd3fc;">-</div>
                </div>
                <div>
                    <div style="font-size: 0.8rem; color: var(--text-muted); margin-bottom: 4px; font-weight: 600;">CONVERSATION CONTEXT</div>
                    <div class="box" id="conversation-context" style="max-height: 120px; overflow-y: auto;">-</div>
                </div>
            </div>

            <div style="margin-top: 10px;">
                <div style="font-size: 0.8rem; color: var(--text-muted); margin-bottom: 4px; font-weight: 600;">POLICY ESCALATION REASON</div>
                <div class="box" id="escalation-reason" style="margin-bottom: 0; color: #fde047;">-</div>
            </div>
        </div>

        <!-- Generated Reply & Historical Case Card -->
        <div class="card">
            <div class="grid-2">
                <div>
                    <div class="card-header" style="color: var(--accent-green);">SYSTEM GENERATED REPLY</div>
                    <div class="box" id="generated-reply" style="font-size: 1rem; border-color: var(--accent-green); background: #064e3b; min-height: 90px;">-</div>
                </div>
                <div>
                    <div class="card-header" style="color: var(--accent-purple);">TOP RETRIEVED HISTORICAL CASE</div>
                    <div class="box" id="retrieved-case" style="border-color: var(--accent-purple); min-height: 90px; max-height: 160px; overflow-y: auto;">-</div>
                </div>
            </div>
        </div>

        <!-- Rating Section Card -->
        <div class="card">
            <div class="card-header">Human Quality Evaluation (1 to 5 Scale)</div>
            
            <div class="rating-section" id="rating-container">
                <!-- Dynamic Rating Boxes generated by JS -->
            </div>

            <div style="margin-top: 16px;">
                <div class="rating-title">Optional Comment / Failure Mode Note</div>
                <textarea class="comment-box" id="annotator-comment" placeholder="Add optional comments regarding errors, phrasing, or routing quality..."></textarea>
            </div>
        </div>

        <!-- Navigation Footer -->
        <div class="nav-controls">
            <button class="nav-btn" id="btn-prev" onclick="navigate(-1)">← Previous</button>
            <div style="display: flex; align-items: center; gap: 16px;">
                <span class="status-toast" id="status-toast">✓ Saved</span>
                <button class="nav-btn nav-btn-primary" id="btn-save" onclick="saveCurrent(true)">Save & Next →</button>
            </div>
            <button class="nav-btn" id="btn-next" onclick="navigate(1)">Next →</button>
        </div>
    </div>

    <script>
        let examples = [];
        let currentIndex = 0;
        let ratings = {}; // example_id -> rating dict

        const ratingDefs = {
            relevance_1to5: {
                title: "1. RELEVANCE",
                desc: "Does the reply address the customer's actual support issue?",
                scale: "1 = Irrelevant | 3 = Partial | 5 = Directly addresses issue"
            },
            groundedness_1to5: {
                title: "2. GROUNDEDNESS",
                desc: "Consistent with retrieved historical AmazonHelp evidence & avoids unsupported claims?",
                scale: "1 = Hallucinates | 3 = Mostly consistent | 5 = Fully grounded"
            },
            helpfulness_1to5: {
                title: "3. HELPFULNESS",
                desc: "Provides actionable & appropriate next steps or routing?",
                scale: "1 = Unhelpful/Dead-end | 3 = Moderate help | 5 = Clear next steps"
            },
            tone_1to5: {
                title: "4. TONE",
                desc: "Professional, concise, empathetic, and suitable for customer support?",
                scale: "1 = Inappropriate/Wordy | 3 = Acceptable | 5 = Empathetic & crisp"
            },
            safety_1to5: {
                title: "5. SAFETY / ESCALATION CONSISTENCY",
                desc: "Avoids claiming private account/order actions publicly? Directs to private channel if needed?",
                scale: "1 = Unsafe/Public order claims | 3 = Safe | 5 = Perfect safe routing"
            },
            overall_quality_1to5: {
                title: "6. OVERALL QUALITY",
                desc: "Overall, would this be a reasonable first-draft response for a human support agent?",
                scale: "1 = Poor (Discard) | 3 = Acceptable (Needs edit) | 5 = Excellent (Ready)"
            }
        };

        async function init() {
            renderRatingBoxes();
            const res = await fetch('/api/data');
            examples = await res.json();
            
            // Populate initial ratings
            examples.forEach(ex => {
                ratings[ex.example_id] = {
                    relevance_1to5: ex.relevance_1to5 || "",
                    groundedness_1to5: ex.groundedness_1to5 || "",
                    helpfulness_1to5: ex.helpfulness_1to5 || "",
                    tone_1to5: ex.tone_1to5 || "",
                    safety_1to5: ex.safety_1to5 || "",
                    overall_quality_1to5: ex.overall_quality_1to5 || "",
                    annotator_comment: ex.annotator_comment || ""
                };
            });

            // Find first incomplete example or default 0
            const firstIncomplete = examples.findIndex(ex => !isComplete(ex.example_id));
            if (firstIncomplete !== -1) {
                currentIndex = firstIncomplete;
            }

            renderCurrent();
        }

        function isComplete(id) {
            const r = ratings[id];
            if (!r) return false;
            return r.relevance_1to5 && r.groundedness_1to5 && r.helpfulness_1to5 && r.tone_1to5 && r.safety_1to5 && r.overall_quality_1to5;
        }

        function renderRatingBoxes() {
            const container = document.getElementById('rating-container');
            container.innerHTML = '';

            Object.keys(ratingDefs).forEach(key => {
                const def = ratingDefs[key];
                const box = document.createElement('div');
                box.className = 'rating-box';
                box.innerHTML = `
                    <div class="rating-title">${def.title}</div>
                    <div class="rating-desc">${def.desc}</div>
                    <div class="scale-guide">${def.scale}</div>
                    <div class="btn-group">
                        ${[1,2,3,4,5].map(v => `
                            <button class="rating-btn" data-key="${key}" data-val="${v}" onclick="selectRating('${key}', ${v})">${v}</button>
                        `).join('')}
                    </div>
                `;
                container.appendChild(box);
            });
        }

        function renderCurrent() {
            if (examples.length === 0) return;
            const ex = examples[currentIndex];
            const currentRatings = ratings[ex.example_id] || {};

            document.getElementById('nav-info').innerText = `Example ID: ${ex.example_id} (${currentIndex + 1} of ${examples.length})`;
            
            // Header stats
            const completedCount = Object.keys(ratings).filter(id => isComplete(id)).length;
            document.getElementById('completion-count').innerText = `${completedCount} / ${examples.length} Completed`;
            document.getElementById('progress-fill').style.width = `${(completedCount / examples.length) * 100}%`;

            // Metadata & context
            document.getElementById('customer-message').innerText = ex.customer_message || '-';
            document.getElementById('conversation-context').innerText = ex.conversation_context || '-';
            document.getElementById('escalation-reason').innerText = ex.policy_escalation_reason || '-';
            document.getElementById('badge-intent').innerText = ex.predicted_intent;

            const isEsc = (ex.final_escalation === 'True' || ex.final_escalation === true);
            const escBadge = document.getElementById('badge-escalation');
            escBadge.innerText = isEsc ? 'ESCALATED (True)' : 'PUBLIC (False)';
            escBadge.className = `badge ${isEsc ? 'badge-true' : 'badge-false'}`;

            // Reply & evidence
            document.getElementById('generated-reply').innerText = ex.generated_reply || '(No public reply text — escalated to private channel)';
            document.getElementById('retrieved-case').innerText = ex.top1_retrieved_case || 'None';

            // Active button selections
            Object.keys(ratingDefs).forEach(key => {
                const val = currentRatings[key];
                document.querySelectorAll(`.rating-btn[data-key="${key}"]`).forEach(btn => {
                    if (parseInt(btn.dataset.val) === parseInt(val)) {
                        btn.classList.add('active');
                    } else {
                        btn.classList.remove('active');
                    }
                });
            });

            // Comment field
            document.getElementById('annotator-comment').value = currentRatings.annotator_comment || '';

            // Button state
            document.getElementById('btn-prev').disabled = (currentIndex === 0);
            document.getElementById('btn-next').disabled = (currentIndex === examples.length - 1);
        }

        function selectRating(key, value) {
            const ex = examples[currentIndex];
            if (!ratings[ex.example_id]) ratings[ex.example_id] = {};
            ratings[ex.example_id][key] = value.toString();

            // Highlight UI immediately
            document.querySelectorAll(`.rating-btn[data-key="${key}"]`).forEach(btn => {
                if (parseInt(btn.dataset.val) === value) {
                    btn.classList.add('active');
                } else {
                    btn.classList.remove('active');
                }
            });

            saveCurrent(false);
        }

        async function saveCurrent(andNext = false) {
            const ex = examples[currentIndex];
            const currentRatings = ratings[ex.example_id] || {};
            currentRatings.annotator_comment = document.getElementById('annotator-comment').value.trim();

            try {
                const res = await fetch('/api/save', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        example_id: ex.example_id,
                        ratings: currentRatings
                    })
                });

                if (res.ok) {
                    showToast();
                    if (andNext && currentIndex < examples.length - 1) {
                        currentIndex++;
                        renderCurrent();
                    } else {
                        renderCurrent();
                    }
                }
            } catch (e) {
                console.error("Error saving rating:", e);
            }
        }

        function showToast() {
            const toast = document.getElementById('status-toast');
            toast.classList.add('show');
            setTimeout(() => toast.classList.remove('show'), 1200);
        }

        function navigate(dir) {
            saveCurrent(false);
            const newIndex = currentIndex + dir;
            if (newIndex >= 0 && newIndex < examples.length) {
                currentIndex = newIndex;
                renderCurrent();
            }
        }

        document.getElementById('annotator-comment').addEventListener('blur', () => saveCurrent(false));

        init();
    </script>
</body>
</html>
"""


def load_template_and_annotations():
    """Load template and merge with existing human_annotations.csv if present."""
    if not os.path.exists(TEMPLATE_PATH):
        raise FileNotFoundError(f"Template not found at {TEMPLATE_PATH}")

    with open(TEMPLATE_PATH, 'r', encoding='utf-8') as f:
        template_rows = list(csv.DictReader(f))

    existing_annotations = {}
    if os.path.exists(ANNOTATIONS_PATH):
        with open(ANNOTATIONS_PATH, 'r', encoding='utf-8') as f:
            for r in csv.DictReader(f):
                existing_annotations[r['example_id']] = r

    fieldnames = list(template_rows[0].keys())

    merged_data = []
    for row in template_rows:
        eid = row['example_id']
        merged_row = dict(row)
        if eid in existing_annotations:
            for k in ['relevance_1to5', 'groundedness_1to5', 'helpfulness_1to5', 'tone_1to5', 'safety_1to5', 'overall_quality_1to5', 'annotator_comment']:
                merged_row[k] = existing_annotations[eid].get(k, '')
        merged_data.append(merged_row)

    return merged_data, fieldnames


def save_single_annotation(example_id: str, new_ratings: dict):
    """Update human_annotations.csv with new rating for given example_id."""
    data, fieldnames = load_template_and_annotations()

    for row in data:
        if str(row['example_id']) == str(example_id):
            for k, v in new_ratings.items():
                if k in fieldnames:
                    row[k] = str(v)

    # Write out safely
    temp_file = ANNOTATIONS_PATH + ".tmp"
    with open(temp_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

    os.replace(temp_file, ANNOTATIONS_PATH)


class AnnotationHandler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        # Quiet standard HTTP logging
        return

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == '/' or parsed.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode('utf-8'))
        elif parsed.path == '/api/data':
            data, _ = load_template_and_annotations()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(data).encode('utf-8'))
        else:
            self.send_error(404, "Page Not Found")

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == '/api/save':
            content_len = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_len).decode('utf-8')
            try:
                payload = json.loads(body)
                example_id = payload.get('example_id')
                ratings = payload.get('ratings', {})
                save_single_annotation(example_id, ratings)
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"status": "ok"}).encode('utf-8'))
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"status": "error", "message": str(e)}).encode('utf-8'))
        else:
            self.send_error(404, "Endpoint Not Found")


def main():
    parser = argparse.ArgumentParser(description="Reply Quality Human Annotation Web App")
    parser.add_argument("--port", type=int, default=8500, help="Port to run server on (default: 8500)")
    args = parser.parse_args()

    # Verify template exists
    if not os.path.exists(TEMPLATE_PATH):
        print(f"Error: Template file missing at {TEMPLATE_PATH}")
        sys.exit(1)

    server_address = ('', args.port)
    httpd = HTTPServer(server_address, AnnotationHandler)

    url = f"http://localhost:{args.port}"
    print("=" * 80)
    print("  HUMAN ANNOTATION TOOL READY")
    print("=" * 80)
    print(f"  URL        : {url}")
    print(f"  Target File: {ANNOTATIONS_PATH}")
    print("=" * 80)
    print("  Press Ctrl+C to stop the server.\n")

    # Try opening browser automatically
    try:
        webbrowser.open(url)
    except Exception:
        pass

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server. Human annotations saved at:", ANNOTATIONS_PATH)


if __name__ == '__main__':
    main()
