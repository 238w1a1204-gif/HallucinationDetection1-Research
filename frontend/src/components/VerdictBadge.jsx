/**
 * Coloured badge for any verdict / status string returned by the API.
 * Covers claim-level verdicts, answer-level hallucination verdicts,
 * and answer-value-check statuses.
 */

const STYLES = {
  // ── Claim-level verdicts ────────────────────────────────────────────────
  VERIFIED:                              'bg-green-100  text-green-800  border border-green-300',
  PARTIALLY_SUPPORTED:                   'bg-yellow-100 text-yellow-800 border border-yellow-300',
  UNSUPPORTED:                           'bg-gray-100   text-gray-700   border border-gray-300',
  CONTRADICTED:                          'bg-red-100    text-red-800    border border-red-300',

  // ── Answer-level hallucination verdicts ─────────────────────────────────
  NO_HALLUCINATION_DETECTED:             'bg-green-100  text-green-800  border border-green-300',
  CONTAINS_HALLUCINATION:                'bg-red-100    text-red-800    border border-red-300',
  PARTIALLY_HALLUCINATED_OR_UNVERIFIABLE:'bg-yellow-100 text-yellow-800 border border-yellow-300',
  NO_CLAIMS_EXTRACTED:                   'bg-gray-100   text-gray-600   border border-gray-300',

  // ── Answer-value check statuses ─────────────────────────────────────────
  VALUE_SUPPORTED:    'bg-green-50  text-green-700  border border-green-200',
  VALUE_CONTRADICTED: 'bg-red-50    text-red-700    border border-red-200',
  VALUE_UNVERIFIABLE: 'bg-gray-50   text-gray-500   border border-gray-200',
};

const LABELS = {
  // Claim-level
  VERIFIED:                              'Verified',
  PARTIALLY_SUPPORTED:                   'Partially Supported',
  UNSUPPORTED:                           'Unsupported',
  CONTRADICTED:                          'Contradicted',

  // Answer-level
  NO_HALLUCINATION_DETECTED:             'No Hallucination Detected',
  CONTAINS_HALLUCINATION:                'Contains Hallucination',
  PARTIALLY_HALLUCINATED_OR_UNVERIFIABLE:'Partially Hallucinated / Unverifiable',
  NO_CLAIMS_EXTRACTED:                   'No Claims Extracted',

  // Answer-value
  VALUE_SUPPORTED:    'Value Supported',
  VALUE_CONTRADICTED: 'Value Contradicted',
  VALUE_UNVERIFIABLE: 'Value Unverifiable',
};

export default function VerdictBadge({ verdict, size = 'sm' }) {
  if (!verdict) return null;
  const styleClass = STYLES[verdict] ?? 'bg-gray-100 text-gray-600 border border-gray-300';
  const label      = LABELS[verdict] ?? verdict.replace(/_/g, ' ');
  const sizeClass  = size === 'lg'
    ? 'px-3 py-1 text-sm font-semibold'
    : 'px-2 py-0.5 text-xs font-medium';

  return (
    <span className={`inline-block rounded-full whitespace-nowrap ${sizeClass} ${styleClass}`}>
      {label}
    </span>
  );
}
