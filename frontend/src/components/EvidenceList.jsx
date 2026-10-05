/**
 * Renders the list of evidence items for a single claim.
 */
export default function EvidenceList({ evidence = [] }) {
  if (!evidence.length) {
    return (
      <p className="text-xs text-gray-400 italic">
        No evidence retrieved from any source.
      </p>
    );
  }

  return (
    <ul className="space-y-2 mt-2">
      {evidence.map((ev) => (
        <li
          key={ev.id}
          className="bg-gray-50 rounded-lg border border-gray-100 p-3 text-xs"
        >
          <div className="flex items-center justify-between mb-1">
            <SourceBadge source={ev.source_name} />
            <span className="text-gray-500">
              similarity:{' '}
              <span className="font-medium text-gray-700">
                {ev.similarity_score != null
                  ? (ev.similarity_score * 100).toFixed(1) + '%'
                  : '—'}
              </span>
            </span>
          </div>
          <p className="text-gray-700 leading-relaxed">{ev.evidence_text}</p>
          {ev.source_url && (
            <a
              href={ev.source_url}
              target="_blank"
              rel="noopener noreferrer"
              className="mt-1 inline-block text-blue-600 hover:underline truncate max-w-full"
            >
              {ev.source_url}
            </a>
          )}
        </li>
      ))}
    </ul>
  );
}

const SOURCE_STYLES = {
  wikipedia: 'bg-blue-50 text-blue-700 border border-blue-200',
  wikidata: 'bg-purple-50 text-purple-700 border border-purple-200',
  duckduckgo: 'bg-orange-50 text-orange-700 border border-orange-200',
};

function SourceBadge({ source }) {
  const style =
    SOURCE_STYLES[source] ?? 'bg-gray-50 text-gray-600 border border-gray-200';
  return (
    <span className={`px-2 py-0.5 rounded-full text-xs font-medium capitalize ${style}`}>
      {source}
    </span>
  );
}
