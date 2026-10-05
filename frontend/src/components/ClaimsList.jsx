import ClaimCard from './ClaimCard';

/**
 * Renders all claims for the current result.
 */
export default function ClaimsList({ claims = [] }) {
  if (!claims.length) {
    return (
      <p className="text-sm text-gray-400 italic">
        No individual claims were extracted from this answer.
      </p>
    );
  }

  return (
    <section>
      <h2 className="text-lg font-semibold text-gray-800 mb-3">
        Claim-by-Claim Breakdown
      </h2>
      <div className="space-y-3">
        {claims.map((claim, i) => (
          <ClaimCard key={claim.id ?? i} claim={claim} index={i} />
        ))}
      </div>
    </section>
  );
}
