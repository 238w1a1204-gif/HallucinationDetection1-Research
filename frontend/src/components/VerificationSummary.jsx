import VerdictBadge from './VerdictBadge';

/**
 * Top-level summary card: overall verdict, confidence, claim counts,
 * and (when available) the question type / relationship.
 */
export default function VerificationSummary({ result }) {
  const {
    overall_verdict,
    overall_confidence,
    claims = [],
    question_type,
    question_relationship,
  } = result;

  const numClaims = claims.length;
  const numVerified = claims.filter(
    (c) => c.verification_result?.verdict === 'VERIFIED'
  ).length;
  const numPartial = claims.filter(
    (c) => c.verification_result?.verdict === 'PARTIALLY_SUPPORTED'
  ).length;
  const numUnsupported = claims.filter(
    (c) => c.verification_result?.verdict === 'UNSUPPORTED'
  ).length;
  const numContradicted = claims.filter(
    (c) => c.verification_result?.verdict === 'CONTRADICTED'
  ).length;

  const confidencePct = Math.round((overall_confidence ?? 0) * 100);

  return (
    <section className="bg-white rounded-xl border border-gray-200 shadow-sm p-5">
      <h2 className="text-lg font-semibold text-gray-800 mb-3">
        Verification Summary
      </h2>

      {/* Question analysis metadata */}
      {(question_type || question_relationship) && (
        <div className="flex flex-wrap gap-2 mb-3">
          {question_type && (
            <span className="px-2 py-0.5 rounded-full text-xs font-medium
                             bg-indigo-50 text-indigo-700 border border-indigo-200">
              Expected answer type: {question_type}
            </span>
          )}
          {question_relationship && (
            <span className="px-2 py-0.5 rounded-full text-xs font-medium
                             bg-violet-50 text-violet-700 border border-violet-200">
              Relationship: {question_relationship}
            </span>
          )}
        </div>
      )}

      <div className="flex flex-wrap items-center gap-3 mb-4">
        <VerdictBadge verdict={overall_verdict} size="lg" />
        <span className="text-sm text-gray-500">
          Overall confidence:{' '}
          <span className="font-medium text-gray-800">{confidencePct}%</span>
          <span className="ml-1 text-xs text-gray-400">
            (heuristic — not a calibrated probability)
          </span>
        </span>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
        <Stat label="Total Claims"       value={numClaims}       color="text-gray-700" />
        <Stat label="Verified"           value={numVerified}     color="text-green-700" />
        <Stat label="Partial"            value={numPartial}      color="text-yellow-700" />
        <Stat label="Unsupported"        value={numUnsupported}  color="text-gray-500" />
        <Stat label="Contradicted"       value={numContradicted} color="text-red-700" />
      </div>
    </section>
  );
}

function Stat({ label, value, color }) {
  return (
    <div className="bg-gray-50 rounded-lg p-3 text-center">
      <div className={`text-2xl font-bold ${color}`}>{value}</div>
      <div className="text-xs text-gray-500 mt-0.5">{label}</div>
    </div>
  );
}
