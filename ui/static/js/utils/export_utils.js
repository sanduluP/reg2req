/**
 * Export Utils 📦
 * --------------
 * Small utilities to export data from the UI as downloadable files (JSON/TXT).
 *
 * Design goals
 * ------------
 * - No server required ✅
 * - Works offline / local ✅
 * - One-liners from controllers ✅
 */

import { formatPredicateLabel } from "./predicate_format.js";

/**
 * @typedef {Object} SubgraphSentenceRecord
 * @property {string} sentence
 * @property {string} [source]
 * @property {number|null} [page_number]
 * @property {string} [relation]
 * @property {string|null} [created_at]
 * @property {string|null} [last_updated_at]
 */

/**
 * Download a JSON object as a `.json` file.
 *
 * @param {Object} opts
 * @param {string} opts.filename
 * @param {any} opts.data
 */
export function downloadJson({ filename, data }) {
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
    _downloadBlob(blob, filename);
}

/**
 * Download a string as a UTF-8 `.txt` file.
 *
 * @param {Object} opts
 * @param {string} opts.filename
 * @param {string} opts.text
 */
export function downloadText({ filename, text }) {
    const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
    _downloadBlob(blob, filename);
}

/**
 * Export subgraph edge sentences as JSON.
 *
 * Output shape:
 * {
 *   keyword: string|null,
 *   exported_at: string,   // ISO timestamp
 *   sentences: Array<SubgraphSentenceRecord>
 * }
 *
 * @param {Object} opts
 * @param {any} opts.subgraphPayload Raw payload from `/api/graph/subgraph`.
 * @param {string|null} opts.keyword Current keyword (for convenience).
 * @param {string} [opts.filename] Optional override filename.
 * @returns {{ ok: true, count: number } | { ok: false, reason: string }}
 */
export function exportSubgraphSentencesAsJson({ subgraphPayload, keyword, filename }) {
    const sentences = _extractSubgraphSentenceRecords(subgraphPayload);

    if (sentences.length === 0) {
        return { ok: false, reason: "No edge sentences found in the current subgraph." };
    }

    const safeKey = _safeSlug(keyword || "keyword");
    const out = {
        keyword: keyword ?? null,
        exported_at: new Date().toISOString(),
        sentences,
    };

    downloadJson({
        filename: filename || `kbdebugger_subgraph_sentences_${safeKey}.json`,
        data: out,
    });

    return { ok: true, count: sentences.length };
}

/**
 * Export subgraph edge sentences as TXT (unique sentences).
 *
 * TXT format:
 * 1. sentence...
 *
 * 2. sentence...
 *
 * @param {Object} opts
 * @param {any} opts.subgraphPayload Raw payload from `/api/graph/subgraph`.
 * @param {string|null} opts.keyword Current keyword (for convenience).
 * @param {string} [opts.filename] Optional override filename.
 * @returns {{ ok: true, count: number } | { ok: false, reason: string }}
 */
export function exportSubgraphSentencesAsTxt({ subgraphPayload, keyword, filename }) {
    const records = _extractSubgraphSentenceRecords(subgraphPayload);
    const uniq = Array.from(new Set(records.map(r => r.sentence).filter(Boolean)));

    if (uniq.length === 0) {
        return { ok: false, reason: "No edge sentences found in the current subgraph." };
    }

    const safeKey = _safeSlug(keyword || "keyword");
    const text = uniq.map((s, i) => `${i + 1}. ${s}`).join("\n\n");

    downloadText({
        filename: filename || `kbdebugger_subgraph_sentences_${safeKey}.txt`,
        text,
    });

    return { ok: true, count: uniq.length };
}

/* ------------------------- internal helpers ------------------------- */

/**
 * Extract sentence records from the server subgraph payload.
 *
 * We prefer `properties.sentence`, but fall back to `original_sentence`.
 *
 * @param {any} payload
 * @returns {SubgraphSentenceRecord[]}
 */
function _extractSubgraphSentenceRecords(payload) {
    const edges = payload?.elements?.edges ?? [];
    /** @type {SubgraphSentenceRecord[]} */
    const out = [];

    for (const e of edges) {
        const props = e?.data?.properties ?? null;
        if (!props) continue;

        const sentence = String(props.sentence || props.original_sentence || "").trim();
        if (!sentence) continue;

        out.push({
            sentence,
            source: props.provenance_source || props.source || "",
            page_number: (props.page_number ?? null),
            relation: e?.data?.label || props.relation || "",
            created_at: props.created_at ?? null,
            last_updated_at: props.last_updated_at ?? null,
        });
    }

    return out;
}

/**
 * Download a Blob via an invisible anchor.
 *
 * @param {Blob} blob
 * @param {string} filename
 */
function _downloadBlob(blob, filename) {
    const url = URL.createObjectURL(blob);

    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();

    URL.revokeObjectURL(url);
}

/**
 * Make a filename-safe slug.
 *
 * @param {string} s
 * @returns {string}
 */
function _safeSlug(s) {
    return String(s)
        .trim()
        .replaceAll(/\s+/g, "_")
        .replaceAll(/[^a-zA-Z0-9_\-]/g, "");
}


/**
 * Export candidate oversight sentences as TXT.
 *
 * Each line corresponds to one table row sentence.
 *
 * TXT format:
 * 1. sentence...
 *
 * 2. sentence...
 *
 * @param {Object} opts
 * @param {string[]} opts.sentences
 * @param {string|null} [opts.keyword]
 * @param {string} [opts.filename]
 * @returns {{ ok: true, count: number } | { ok: false, reason: string }}
 */
export function exportSentencesAsTxt({ sentences, keyword, filename }) {
    const clean = Array.from(
        new Set(
            (sentences || [])
                .map(s => String(s || "").trim())
                .filter(Boolean)
        )
    );

    if (clean.length === 0) {
        return { ok: false, reason: "No sentences found to export." };
    }

    const safeKey = _safeSlug(keyword || "keyword");
    const text = clean.map((s, i) => `${i + 1}. ${s}`).join("\n\n");

    downloadText({
        filename: filename || `kbdebugger_candidate_sentences_${safeKey}.txt`,
        text,
    });

    return { ok: true, count: clean.length };
}

/**
 * Export candidate oversight sentences grouped by decision.
 *
 * TXT format:
 * === NEW ===
 * 1. sentence
 *
 * === PARTIALLY NEW ===
 * 1. sentence
 *
 * === EXISTING ===
 * 1. sentence
 *
 * @param {Object} opts
 * @param {Object} opts.grouped
 * @param {string|null} [opts.keyword]
 * @param {string} [opts.filename]
 * @returns {{ ok: true, count: number } | { ok: false, reason: string }}
 */
export function exportGroupedSentencesAsTxt({ grouped, keyword, filename }) {
    if (!grouped) {
        return { ok: false, reason: "No sentences available." };
    }

    const order = ["NEW", "PARTIALLY_NEW", "EXISTING"];

    const sections = [];

    for (const key of order) {
        const items = grouped[key] || [];
        const sentences = Array.from(
            new Set(
                items
                    .map(r => String(r?.quality || "").trim())
                    .filter(Boolean)
            )
        );

        if (sentences.length === 0) continue;

        const header = key.replaceAll("_", " ");
        const body = sentences
            .map((s, i) => `${i + 1}. ${s}`)
            .join("\n\n");

        sections.push(`=== ${header} ===\n\n${body}`);
    }

    if (sections.length === 0) {
        return { ok: false, reason: "No sentences found to export." };
    }

    const text = sections.join("\n\n\n");

    const safeKey = _safeSlug(keyword || "keyword");

    downloadText({
        filename: filename || `kbdebugger_candidate_sentences_${safeKey}.txt`,
        text,
    });

    return { ok: true, count: sections.length };
}


/**
 * Export candidate oversight rows into ONE Excel sheet.
 *
 * Columns:
 * - Group
 * - Sentence
 * - Similarity
 *
 * @param {Object} opts
 * @param {Object} opts.grouped
 * @param {string|null} [opts.keyword]
 * @param {string} [opts.filename]
 * @returns {{ ok: true, count: number } | { ok: false, reason: string }}
 */
export function exportGroupedSentencesAsXlsx({ grouped, keyword, filename }) {
    if (!grouped) {
        return { ok: false, reason: "No candidate sentences available." };
    }

    const XLSX_LIB = globalThis.XLSX;
    if (!XLSX_LIB) {
        return { ok: false, reason: "XLSX export is not available because SheetJS is not loaded." };
    }

    const orderedGroups = [
        { key: "NEW", label: "New" },
        { key: "PARTIALLY_NEW", label: "Partially New" },
        { key: "EXISTING", label: "Existing" },
    ];

    /** @type {{Group: string, Sentence: string, Similarity: number}[]} */
    const rows = [];

    for (const group of orderedGroups) {
        const items = dedupeRowsBySentenceKeepBestScore(grouped[group.key] || []);

        for (const r of items) {
            const sentence = String(r?.quality || "").trim();
            if (!sentence) continue;

            rows.push({
                Group: group.label,
                Sentence: sentence,
                Similarity: Number((r?.max_score ?? 0).toFixed(2)),
            });
        }
    }

    if (rows.length === 0) {
        return { ok: false, reason: "No candidate sentences found to export." };
    }

    const wb = XLSX_LIB.utils.book_new();
    const ws = XLSX_LIB.utils.json_to_sheet(rows);

    XLSX_LIB.utils.book_append_sheet(wb, ws, "Candidate Sentences");

    const safeKey = _safeSlug(keyword || "keyword");
    const outName = filename || `kbdebugger_candidate_sentences_${safeKey}.xlsx`;

    XLSX_LIB.writeFile(wb, outName);

    return { ok: true, count: rows.length };
}


/**
 * Generic: export an array of plain row objects as a single-sheet xlsx.
 * Column order follows the keys of the first row.
 *
 * @param {Object} opts
 * @param {Object[]} opts.rows
 * @param {string} opts.sheetName
 * @param {string} opts.filename
 * @returns {{ ok: true, count: number } | { ok: false, reason: string }}
 */
export function exportRowsAsXlsx({ rows, sheetName, filename }) {
    const XLSX_LIB = globalThis.XLSX;
    if (!XLSX_LIB) {
        return { ok: false, reason: "XLSX export is not available because SheetJS is not loaded." };
    }
    if (!Array.isArray(rows) || rows.length === 0) {
        return { ok: false, reason: "No rows to export." };
    }

    const wb = XLSX_LIB.utils.book_new();
    const ws = XLSX_LIB.utils.json_to_sheet(rows);
    if (ws["!ref"]) ws["!autofilter"] = { ref: ws["!ref"] };

    XLSX_LIB.utils.book_append_sheet(wb, ws, worksheetName(sheetName || "Export"));
    XLSX_LIB.writeFile(wb, filename || "kbdebugger_export.xlsx");

    return { ok: true, count: rows.length };
}


/**
 * Export extracted triplets for human review into one Excel workbook.
 *
 * @param {Object} opts
 * @param {any[]} opts.rows Current editable triplet rows from the review UI.
 * @param {string|null} [opts.documentName] Original uploaded document filename.
 * @param {string|null} [opts.keyword] Current keyword, used only as a filename fallback.
 * @param {string} [opts.filename]
 * @returns {{ ok: true, count: number, filename: string } | { ok: false, reason: string }}
 */
export function exportTripletReviewAsXlsx({ rows, documentName, keyword, filename }) {
    const XLSX_LIB = globalThis.XLSX;
    if (!XLSX_LIB) {
        return { ok: false, reason: "XLSX export is not available because SheetJS is not loaded." };
    }

    const exportRows = buildTripletReviewRows(rows || []);
    if (exportRows.length === 0) {
        return { ok: false, reason: "No non-deleted extracted triplets found to export." };
    }

    const headers = [
        "Document",
        "Source Chunk",
        "Original Quality",
        "Nearest KG Match",
        "Similarity Score",
        "Extracted Triplet",
        "Faithfulness (1-3)",
        "Relevance (1-3)",
        "Completeness (1-3)",
    ];
    const aoa = [
        headers,
        ...exportRows.map(row => headers.map(header => row[header] ?? "")),
    ];

    const wb = XLSX_LIB.utils.book_new();
    const ws = XLSX_LIB.utils.aoa_to_sheet(aoa);

    addReviewHeaderComments(ws);
    applyReviewSheetStyles(XLSX_LIB, ws);
    ws["!cols"] = [
        { wch: 28 },
        { wch: 72 },
        { wch: 54 },
        { wch: 54 },
        { wch: 16 },
        { wch: 64 },
        { wch: 18 },
        { wch: 16 },
        { wch: 18 },
    ];

    if (ws["!ref"]) {
        ws["!autofilter"] = { ref: ws["!ref"] };
    }

    const baseName = documentBaseName(documentName || "");
    const sheetName = worksheetName(baseName || "Triplet Review");
    const outName = filename || tripletReviewFilename({ documentName, keyword });

    XLSX_LIB.utils.book_append_sheet(wb, ws, sheetName);
    XLSX_LIB.writeFile(wb, outName, { cellStyles: true });

    return { ok: true, count: exportRows.length, filename: outName };
}


function buildTripletReviewRows(rows) {
    return rows
        .filter(row => row && !row.deleted)
        .map((row, index) => ({
            row,
            index,
            sourceIndex: sourceChunkIndex(row),
            quality: String(row?.originalQuality || row?.sentence || "").trim(),
        }))
        .filter(item => {
            const row = item.row;
            return Boolean(
                String(row?.subject || "").trim()
                && String(row?.predicate || "").trim()
                && String(row?.object || "").trim()
            );
        })
        .sort((a, b) => {
            const sourceA = Number.isFinite(a.sourceIndex) ? a.sourceIndex : Number.MAX_SAFE_INTEGER;
            const sourceB = Number.isFinite(b.sourceIndex) ? b.sourceIndex : Number.MAX_SAFE_INTEGER;
            if (sourceA !== sourceB) return sourceA - sourceB;

            const qualityCmp = a.quality.localeCompare(b.quality);
            if (qualityCmp !== 0) return qualityCmp;

            return a.index - b.index;
        })
        .map(({ row }) => ({
            "Document": String(row?.docName || row?.sourceContext?.doc_name || row?.sourceContext?.metadata?.source || "").trim(),
            "Source Chunk": sourceChunkText(row),
            "Original Quality": String(row?.originalQuality || row?.sentence || "").trim(),
            "Nearest KG Match": String(row?.matchedNeighborSentence || "").trim(),
            "Similarity Score": Number.isFinite(row?.maxScore) ? Number(row.maxScore.toFixed(3)) : "",
            "Extracted Triplet": tripletText(row),
            "Faithfulness (1-3)": "",
            "Relevance (1-3)": "",
            "Completeness (1-3)": "",
        }));
}


function addReviewHeaderComments(ws) {
    const comments = {
        G1: "3 = Fully supported by source chunk; 2 = Partially supported / minor nuance missing; 1 = Unsupported / hallucinated.",
        H1: "3 = Directly relevant to selected Trustworthy AI topic; 2 = Partially / indirectly relevant; 1 = Irrelevant.",
        I1: "3 = Context preserved; 2 = Some context missing; 1 = Important context lost / misleading.",
    };

    for (const [cellRef, text] of Object.entries(comments)) {
        if (!ws[cellRef]) continue;
        ws[cellRef].c = [{ a: "KBDebugger", t: text, hidden: true, h: 1 }];
    }
}


function applyReviewSheetStyles(XLSX_LIB, ws) {
    if (!ws["!ref"] || !XLSX_LIB?.utils?.decode_range || !XLSX_LIB?.utils?.encode_cell) return;

    const range = XLSX_LIB.utils.decode_range(ws["!ref"]);
    const wrappedColumns = new Set([1, 2, 3, 5]);

    for (let row = range.s.r; row <= range.e.r; row += 1) {
        for (let col = range.s.c; col <= range.e.c; col += 1) {
            const ref = XLSX_LIB.utils.encode_cell({ r: row, c: col });
            const cell = ws[ref];
            if (!cell) continue;

            const shouldWrap = wrappedColumns.has(col) || row === 0;
            cell.s = {
                ...(cell.s || {}),
                alignment: {
                    ...((cell.s && cell.s.alignment) || {}),
                    vertical: "top",
                    ...(shouldWrap ? { wrapText: true } : {}),
                },
            };
        }
    }

    ws["!rows"] = [
        { hpt: 28 },
        ...Array.from({ length: Math.max(0, range.e.r) }, () => ({ hpt: 54 })),
    ];
}


function sourceChunkIndex(row) {
    const raw = row?.sourceContext?.source_doc_index;
    const parsed = Number(raw);
    return Number.isFinite(parsed) ? parsed : null;
}


function sourceChunkText(row) {
    const ctx = row?.sourceContext || {};
    const metadata = ctx?.metadata || {};
    const headings = Array.isArray(metadata?.headings)
        ? metadata.headings.map(String).filter(Boolean)
        : [];
    const text = String(ctx?.source_text || "").trim();
    const parts = [];
    if (headings.length > 0) parts.push(headings.join(" > "));
    if (text) parts.push(text);
    return parts.join("\n\n");
}


function tripletText(row) {
    const subject = String(row?.subject || "").trim();
    const predicate = formatPredicateLabel(row?.predicate || "");
    const object = String(row?.object || "").trim();
    return `${subject} --${predicate}--> ${object}`;
}


function tripletReviewFilename({ documentName, keyword }) {
    const baseName = documentBaseName(documentName || "");
    const safeDocument = _safeSlug(baseName);
    if (safeDocument) return `${safeDocument}_triplet_review.xlsx`;

    const safeKey = _safeSlug(keyword || "keyword") || "keyword";
    return `kbdebugger_triplet_review_${safeKey}.xlsx`;
}


function documentBaseName(name) {
    const filename = String(name || "").split(/[\\/]/).pop().trim();
    if (!filename) return "";
    return filename.replace(/\.[^/.]+$/, "").trim();
}


function worksheetName(name) {
    const cleaned = String(name || "")
        .replaceAll(/[:\\/?*\[\]]/g, " ")
        .replaceAll(/\s+/g, " ")
        .trim()
        .slice(0, 31)
        .trim();

    return cleaned || "Triplet Review";
}

/**
 * Deduplicate rows by sentence text, keeping the highest-similarity row.
 *
 * @param {any[]} items
 * @returns {any[]}
 */
function dedupeRowsBySentenceKeepBestScore(items) {
    const bestByKey = new Map();

    for (const r of items || []) {
        const sentence = String(r?.quality || "").trim();
        const key = sentence.toLowerCase();
        if (!sentence) continue;

        const existing = bestByKey.get(key);
        if (!existing || (r?.max_score ?? 0) > (existing?.max_score ?? 0)) {
            bestByKey.set(key, r);
        }
    }

    return Array.from(bestByKey.values());
}
