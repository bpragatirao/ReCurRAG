/**
 * ReCurRAG Dashboard — JavaScript Application
 * 
 * Loads the comparison_report.json and renders:
 *   1. Overall summary cards
 *   2. Per-dataset metric bars
 *   3. Detailed Q&A comparison with side-by-side answers
 */

document.addEventListener('DOMContentLoaded', init);

let reportData = null;

async function init() {
    try {
        const response = await fetch('comparison_report.json');
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        reportData = await response.json();

        document.getElementById('loading').style.display = 'none';
        document.getElementById('dashboard').style.display = 'block';

        renderReportDate();
        renderSummaryCards();
        renderDatasetSections();
        renderQATabs();
    } catch (err) {
        console.error('Failed to load report:', err);
        document.getElementById('loading').style.display = 'none';
        document.getElementById('error').style.display = 'block';
    }
}

/* ---- Report Date ---- */
function renderReportDate() {
    const dateEl = document.getElementById('report-date');
    if (reportData.generated_at) {
        const d = new Date(reportData.generated_at);
        dateEl.textContent = d.toLocaleDateString('en-US', {
            year: 'numeric', month: 'short', day: 'numeric',
            hour: '2-digit', minute: '2-digit'
        });
    }
}

/* ---- Summary Cards ---- */
function renderSummaryCards() {
    const container = document.getElementById('summary-cards');
    const s = reportData.overall_summary || {};
    const rag = s.rag || {};
    const rlm = s.rlm || {};

    const cards = [
        {
            label: 'Datasets Evaluated',
            value: s.total_datasets_evaluated || 0,
            sub: 'across 3 data types',
            cls: 'neutral-card',
            colorCls: 'blue-color'
        },
        {
            label: 'RAG Avg Quality',
            value: fmtPct(rag.avg_quality),
            sub: 'answer quality score',
            cls: 'rag-card',
            colorCls: 'rag-color'
        },
        {
            label: 'RLM Avg Quality',
            value: fmtPct(rlm.avg_quality),
            sub: 'answer quality score',
            cls: 'rlm-card',
            colorCls: 'rlm-color'
        },
        {
            label: 'RAG Avg Latency',
            value: `${(rag.avg_latency_s || 0).toFixed(2)}s`,
            sub: 'per query',
            cls: 'rag-card',
            colorCls: 'rag-color'
        },
        {
            label: 'RLM Avg Latency',
            value: `${(rlm.avg_latency_s || 0).toFixed(2)}s`,
            sub: 'per query',
            cls: 'rlm-card',
            colorCls: 'rlm-color'
        },
        {
            label: 'RLM Reasoning Depth',
            value: (rlm.avg_reasoning_depth || 0).toFixed(1),
            sub: 'avg tool calls + reasoning',
            cls: 'rlm-card',
            colorCls: 'rlm-color'
        },
    ];

    // Add EM/F1 cards if available
    if (rag.avg_exact_match !== undefined) {
        cards.push({
            label: 'RAG Exact Match',
            value: fmtPct(rag.avg_exact_match),
            sub: 'multi-hop QA',
            cls: 'rag-card',
            colorCls: 'rag-color'
        });
        cards.push({
            label: 'RLM Exact Match',
            value: fmtPct(rlm.avg_exact_match),
            sub: 'multi-hop QA',
            cls: 'rlm-card',
            colorCls: 'rlm-color'
        });
    }

    container.innerHTML = cards.map(c => `
        <div class="summary-card ${c.cls}">
            <div class="card-label">${c.label}</div>
            <div class="card-value ${c.colorCls}">${c.value}</div>
            <div class="card-sub">${c.sub}</div>
        </div>
    `).join('');
}

/* ---- Per-Dataset Sections ---- */
function renderDatasetSections() {
    const container = document.getElementById('dataset-sections');
    const datasets = reportData.datasets || {};
    let html = '';

    for (const [key, ds] of Object.entries(datasets)) {
        const cfg = ds.config || {};
        const status = ds.status || 'unknown';
        const statusCls = status === 'complete' ? 'status-complete' : 'status-incomplete';

        html += `<section class="dataset-section" id="ds-${key}">`;
        html += `<div class="dataset-header">
            <div>
                <div class="dataset-title">${cfg.icon || '📁'} ${cfg.display_name || key}</div>
                <div class="dataset-desc">${cfg.description || ''}</div>
            </div>
            <span class="dataset-status ${statusCls}">${status.toUpperCase()}</span>
        </div>`;

        if (status === 'complete' && ds.metrics) {
            html += renderDatasetMetrics(key, ds);
        } else {
            html += `<p style="color:var(--text-muted);font-size:0.85rem;">⚠️ ${ds.error || 'Data unavailable'}</p>`;
        }

        html += `</section>`;
    }

    container.innerHTML = html;

    // Animate bars after render
    requestAnimationFrame(() => {
        document.querySelectorAll('.bar-fill').forEach(bar => {
            const w = bar.dataset.width;
            bar.style.width = w + '%';
        });
    });
}

function renderDatasetMetrics(key, ds) {
    const agg = ds.metrics.aggregate || {};
    const rag = agg.rag || {};
    const rlm = agg.rlm || {};

    const metrics = [];

    // Quality
    metrics.push({
        label: 'Answer Quality',
        ragVal: rag.avg_quality || 0,
        rlmVal: rlm.avg_quality || 0,
        ragDisplay: fmtPct(rag.avg_quality),
        rlmDisplay: fmtPct(rlm.avg_quality),
        maxVal: 1,
        higherBetter: true,
    });

    // Latency
    const maxLat = Math.max(rag.avg_latency_s || 1, rlm.avg_latency_s || 1) * 1.2;
    metrics.push({
        label: 'Avg Latency',
        ragVal: rag.avg_latency_s || 0,
        rlmVal: rlm.avg_latency_s || 0,
        ragDisplay: (rag.avg_latency_s || 0).toFixed(2) + 's',
        rlmDisplay: (rlm.avg_latency_s || 0).toFixed(2) + 's',
        maxVal: maxLat,
        higherBetter: false,
    });

    // Reasoning Depth
    const maxDepth = Math.max(rlm.avg_reasoning_depth || 1, 5);
    metrics.push({
        label: 'Reasoning Depth',
        ragVal: 0,
        rlmVal: rlm.avg_reasoning_depth || 0,
        ragDisplay: 'N/A',
        rlmDisplay: (rlm.avg_reasoning_depth || 0).toFixed(1),
        maxVal: maxDepth,
        higherBetter: true,
    });

    // Tool Calls
    const maxTools = Math.max(rlm.avg_tool_calls || 1, 8);
    metrics.push({
        label: 'Avg Tool Calls (RLM)',
        ragVal: 0,
        rlmVal: rlm.avg_tool_calls || 0,
        ragDisplay: 'N/A',
        rlmDisplay: (rlm.avg_tool_calls || 0).toFixed(1),
        maxVal: maxTools,
        higherBetter: true,
    });

    // EM & F1 if available
    if (rag.exact_match !== undefined) {
        metrics.push({
            label: 'Exact Match',
            ragVal: rag.exact_match || 0,
            rlmVal: rlm.exact_match || 0,
            ragDisplay: fmtPct(rag.exact_match),
            rlmDisplay: fmtPct(rlm.exact_match),
            maxVal: 1,
            higherBetter: true,
        });
        metrics.push({
            label: 'F1 Score',
            ragVal: rag.f1_score || 0,
            rlmVal: rlm.f1_score || 0,
            ragDisplay: (rag.f1_score || 0).toFixed(4),
            rlmDisplay: (rlm.f1_score || 0).toFixed(4),
            maxVal: 1,
            higherBetter: true,
        });
    }

    let html = `<div class="metrics-grid">`;

    for (const m of metrics) {
        const ragPct = m.maxVal > 0 ? (m.ragVal / m.maxVal) * 100 : 0;
        const rlmPct = m.maxVal > 0 ? (m.rlmVal / m.maxVal) * 100 : 0;

        let winnerHtml = '';
        if (m.ragVal !== 0 || m.rlmVal !== 0) {
            if (m.higherBetter) {
                if (m.rlmVal > m.ragVal) winnerHtml = '<span class="winner-indicator winner-rlm">RLM ✓</span>';
                else if (m.ragVal > m.rlmVal) winnerHtml = '<span class="winner-indicator winner-rag">RAG ✓</span>';
            } else {
                if (m.ragVal > 0 && m.ragVal < m.rlmVal) winnerHtml = '<span class="winner-indicator winner-rag">RAG ✓</span>';
                else if (m.rlmVal > 0 && m.rlmVal < m.ragVal) winnerHtml = '<span class="winner-indicator winner-rlm">RLM ✓</span>';
            }
        }

        html += `
        <div class="metric-item">
            <div class="metric-label">${m.label} ${winnerHtml}</div>
            <div class="metric-bars">
                <div class="metric-bar-row">
                    <span class="bar-label rag">RAG</span>
                    <div class="bar-track">
                        <div class="bar-fill rag-fill" data-width="${Math.min(ragPct, 100).toFixed(1)}" style="width: 0%"></div>
                    </div>
                    <span class="bar-value">${m.ragDisplay}</span>
                </div>
                <div class="metric-bar-row">
                    <span class="bar-label rlm">RLM</span>
                    <div class="bar-track">
                        <div class="bar-fill rlm-fill" data-width="${Math.min(rlmPct, 100).toFixed(1)}" style="width: 0%"></div>
                    </div>
                    <span class="bar-value">${m.rlmDisplay}</span>
                </div>
            </div>
        </div>`;
    }

    html += `</div>`;
    return html;
}

/* ---- Q&A Tabs & Content ---- */
function renderQATabs() {
    const tabsContainer = document.getElementById('qa-tabs');
    const datasets = reportData.datasets || {};
    let first = true;
    let tabHtml = '';

    for (const [key, ds] of Object.entries(datasets)) {
        if (ds.status !== 'complete') continue;
        const cfg = ds.config || {};
        const activeCls = first ? ' active' : '';
        tabHtml += `<button class="tab-btn${activeCls}" data-dataset="${key}">${cfg.icon || ''} ${cfg.display_name || key}</button>`;
        first = false;
    }

    tabsContainer.innerHTML = tabHtml;

    // Tab click handlers
    tabsContainer.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            tabsContainer.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            renderQAContent(btn.dataset.dataset);
        });
    });

    // Render first tab
    const firstKey = Object.keys(datasets).find(k => datasets[k].status === 'complete');
    if (firstKey) renderQAContent(firstKey);
}

function renderQAContent(datasetKey) {
    const container = document.getElementById('qa-content');
    const ds = reportData.datasets[datasetKey];
    if (!ds || !ds.metrics) {
        container.innerHTML = '<p style="color:var(--text-muted);">No data available.</p>';
        return;
    }

    const queries = ds.metrics.per_query || [];
    let html = '';

    for (const q of queries) {
        const ragQualityCls = qualityClass(q.rag_quality);
        const rlmQualityCls = qualityClass(q.rlm_quality);

        html += `<div class="qa-card">`;
        html += `<div class="qa-question">${escapeHtml(q.question)}</div>`;

        html += `<div class="qa-answers">`;

        // RAG answer
        html += `<div class="qa-answer rag-answer">
            <div class="qa-answer-header">
                <span class="qa-answer-label">RAG Answer</span>
                <span class="qa-answer-meta">${q.rag_latency_s?.toFixed(2) || '?'}s</span>
            </div>
            <div class="qa-answer-text">${escapeHtml(truncate(q.rag_answer, 500))}</div>
        </div>`;

        // RLM answer
        html += `<div class="qa-answer rlm-answer">
            <div class="qa-answer-header">
                <span class="qa-answer-label">RLM Answer</span>
                <span class="qa-answer-meta">${q.rlm_latency_s?.toFixed(2) || '?'}s · ${q.rlm_tool_calls || 0} tools · depth ${q.rlm_reasoning_depth || 0}</span>
            </div>
            <div class="qa-answer-text">${escapeHtml(truncate(q.rlm_answer, 500))}</div>
        </div>`;

        html += `</div>`; // qa-answers

        // Ground truth (if available)
        if (q.ground_truth) {
            html += `<div class="qa-ground-truth">
                <strong>Ground Truth: </strong>${escapeHtml(q.ground_truth)}`;
            if (q.rag_em !== undefined) {
                html += ` · <span style="color:var(--accent-rag);">RAG EM: ${q.rag_em ? '✅' : '❌'}</span>`;
                html += ` · <span style="color:var(--accent-rlm);">RLM EM: ${q.rlm_em ? '✅' : '❌'}</span>`;
            }
            html += `</div>`;
        }

        // Quality badges
        html += `<div class="qa-quality-badges">
            <span class="quality-badge ${ragQualityCls}">RAG: ${fmtPct(q.rag_quality)}</span>
            <span class="quality-badge ${rlmQualityCls}">RLM: ${fmtPct(q.rlm_quality)}</span>
        </div>`;

        html += `</div>`; // qa-card
    }

    container.innerHTML = html;
}

/* ---- Utilities ---- */
function fmtPct(val) {
    if (val === undefined || val === null) return 'N/A';
    return (val * 100).toFixed(0) + '%';
}

function qualityClass(val) {
    if (val >= 0.6) return 'quality-good';
    if (val >= 0.3) return 'quality-ok';
    return 'quality-bad';
}

function truncate(str, maxLen) {
    if (!str) return '(no answer)';
    return str.length > maxLen ? str.substring(0, maxLen) + '...' : str;
}

function escapeHtml(str) {
    if (!str) return '';
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}
