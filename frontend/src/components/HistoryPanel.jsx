import { useEffect, useState } from 'react';
import { listQuestions } from '../api';
import VerdictBadge from './VerdictBadge';

/**
 * Shows the most recent questions from the history endpoint.
 * Clicking a row calls onSelect(question_id).
 */
export default function HistoryPanel({ onSelect, refreshSignal }) {
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    setLoading(true);
    listQuestions({ limit: 20 })
      .then(setItems)
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [refreshSignal]);

  if (loading) {
    return <p className="text-xs text-gray-400 animate-pulse">Loading history…</p>;
  }

  if (!items.length) {
    return (
      <p className="text-xs text-gray-400 italic">
        No previous questions yet.
      </p>
    );
  }

  return (
    <ul className="divide-y divide-gray-100">
      {items.map((item) => (
        <li key={item.question_id}>
          <button
            onClick={() => onSelect(item.question_id)}
            className="w-full text-left px-3 py-2 hover:bg-gray-50 transition-colors"
          >
            <p className="text-sm text-gray-700 truncate">{item.question}</p>
            <div className="flex items-center gap-2 mt-0.5">
              <span className="text-xs text-gray-400">
                {new Date(item.created_at).toLocaleString()}
              </span>
              {item.overall_verdict && (
                <VerdictBadge verdict={item.overall_verdict} />
              )}
            </div>
          </button>
        </li>
      ))}
    </ul>
  );
}
