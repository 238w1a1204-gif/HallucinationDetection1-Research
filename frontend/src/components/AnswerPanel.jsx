/**
 * Displays the original LLM answer and the evidence-corrected answer.
 *
 * The corrected answer string from the API contains two sections separated
 * by "\n\n[Corrections applied]\n":
 *   1. The corrected body text
 *   2. A bullet-point list of what was changed/removed
 *
 * We split them so the UI can display them differently.
 */
export default function AnswerPanel({ answer, correctedAnswer }) {
  const SPLIT = '[Corrections applied]';
  const parts        = (correctedAnswer || '').split(SPLIT);
  const corrBody     = parts[0].trim();
  const corrNotes    = parts[1]?.trim() ?? null;

  const isUnverified = corrBody.toLowerCase().includes('could not be verified')
    || corrBody.toLowerCase().includes('insufficient');

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      {/* ── Original answer ─────────────────────────────────────── */}
      <section className="bg-white rounded-xl border border-gray-200 shadow-sm p-5">
        <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">
          Original LLM Answer
        </h2>
        <p className="text-sm text-gray-800 leading-relaxed whitespace-pre-wrap">
          {answer || <em className="text-gray-400">No answer generated.</em>}
        </p>
      </section>

      {/* ── Corrected answer ────────────────────────────────────── */}
      <section
        className={`rounded-xl border shadow-sm p-5 ${
          isUnverified
            ? 'bg-orange-50 border-orange-200'
            : 'bg-green-50 border-green-200'
        }`}
      >
        <h2
          className={`text-xs font-semibold uppercase tracking-wide mb-2 ${
            isUnverified ? 'text-orange-700' : 'text-green-700'
          }`}
        >
          {isUnverified ? 'Verification Result' : 'Evidence-Corrected Answer'}
        </h2>

        {corrBody ? (
          <p className="text-sm text-gray-800 leading-relaxed whitespace-pre-wrap">
            {corrBody}
          </p>
        ) : (
          <em className="text-sm text-gray-400">No corrected answer available.</em>
        )}

        {/* Corrections log */}
        {corrNotes && (
          <details className="mt-3">
            <summary className="text-xs text-gray-500 cursor-pointer hover:text-gray-700">
              Corrections applied ▸
            </summary>
            <pre className="mt-2 text-xs text-gray-600 whitespace-pre-wrap leading-relaxed
                            bg-white bg-opacity-60 rounded p-2 border border-gray-100">
              {corrNotes}
            </pre>
          </details>
        )}
      </section>
    </div>
  );
}
