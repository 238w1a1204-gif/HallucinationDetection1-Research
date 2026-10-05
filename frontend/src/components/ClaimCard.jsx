import { useState } from 'react';
import VerdictBadge from './VerdictBadge';
import EvidenceList from './EvidenceList';

// Human-readable labels for the answer-value check status
const AV_LABELS = {
  VALUE_SUPPORTED:    { text: '✓ value supported',      cls: 'text-green-600' },
  VALUE_CONTRADICTED: { text: '✗ value contradicted',   cls: 'text-red-600' },
  VALUE_UNVERIFIABLE: { text: '? value unverifiable',   cls: 'text-gray-400' },
};

/**
 * Card for a single extracted claim.
 * Shows: claim text, verdict badge, confidence, answer-value status,
 * explanation, and collapsible evidence.
 */
export default function ClaimCard({ claim, index }) {
  const [expanded, setExpanded] = useState(false);
  const vr = claim.verification_result;
  const confidencePct = claim.confidence != null
    ? Math.round(claim.confidence * 100)
    : null;

  // answer_value_status is surfaced in the explanation string; parse it out
  // so we can display it as a badge separately.
  const avMatch = vr?.explanation?.match(/Answer-value[^\n.]*?(VALUE_[A-Z_]+)/);
  const avStatus = avMatch ? avMatch[1] : null;
  const avLabel  = avStatus ? AV_LABELS[avStatus] : null;

  return (
    <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
      {/* Header */}
      <div className="p-4 flex items-start gap-3">
        <span className="mt-0.5 flex-shrink-0 w-6 h-6 rounded-full bg-gray-100
                         text-gray-500 text-xs font-bold flex items-center justify-center">
          {index + 1}
        </span>

        <div className="flex-1 min-w-0">
          {/* Claim text */}
          <p className="text-sm font-medium text-gray-800 leading-snug">
            &ldquo;{claim.claim_text}&rdquo;
          </p>

          {/* Badges row */}
          <div className="flex flex-wrap items-center gap-2 mt-2">
            {vr && <VerdictBadge verdict={vr.verdict} />}

            {confidencePct != null && (
              <span className="text-xs text-gray-500">
                confidence:{' '}
                <span className="font-medium text-gray-700">{confidencePct}%</span>
              </span>
            )}

            {/* Answer-value check result */}
            {avLabel && (
              <span className={`text-xs font-medium ${avLabel.cls}`}>
                {avLabel.text}
              </span>
            )}

            {/* Entity / NLI flags */}
            {vr?.contradiction_detected && (
              <span className="text-xs text-red-500">⚠ NLI contradiction</span>
            )}
            {vr?.entity_mismatch && (
              <span className="text-xs text-red-500">⚠ entity/date mismatch</span>
            )}
          </div>

          {/* Explanation (collapsed to 3 lines, expandable) */}
          {vr?.explanation && (
            <ExplanationText text={vr.explanation} />
          )}
        </div>
      </div>

      {/* Evidence toggle */}
      {claim.evidence && claim.evidence.length > 0 && (
        <div className="border-t border-gray-100">
          <button
            onClick={() => setExpanded((v) => !v)}
            className="w-full flex items-center justify-between px-4 py-2
                       text-xs text-gray-500 hover:bg-gray-50 transition-colors"
            aria-expanded={expanded}
          >
            <span>
              Evidence ({claim.evidence.length} source
              {claim.evidence.length !== 1 ? 's' : ''})
            </span>
            <span className="text-gray-400">{expanded ? '▲' : '▼'}</span>
          </button>

          {expanded && (
            <div className="px-4 pb-4">
              <EvidenceList evidence={claim.evidence} />
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/** Shows explanation text; long text gets a "show more" toggle. */
function ExplanationText({ text }) {
  const [open, setOpen] = useState(false);
  const SHORT = 180;
  const isLong = text.length > SHORT;

  return (
    <div className="mt-2 text-xs text-gray-500 leading-relaxed">
      {open || !isLong ? text : text.slice(0, SHORT) + '…'}
      {isLong && (
        <button
          onClick={() => setOpen((v) => !v)}
          className="ml-1 text-blue-500 hover:underline"
        >
          {open ? 'show less' : 'show more'}
        </button>
      )}
    </div>
  );
}
