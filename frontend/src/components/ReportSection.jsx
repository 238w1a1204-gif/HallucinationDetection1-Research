import { useState } from 'react';
import { getReport } from '../api';

/**
 * Loads and displays the full verification report for a question.
 * Also provides a "Download as text" button.
 */
export default function ReportSection({ questionId }) {
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showJson, setShowJson] = useState(false);

  async function fetchReport() {
    setLoading(true);
    setError(null);
    try {
      const data = await getReport(questionId);
      setReport(data);
    } catch (err) {
      setError(err?.response?.data?.detail ?? err.message ?? 'Failed to load report.');
    } finally {
      setLoading(false);
    }
  }

  function downloadText() {
    if (!report?.report_text) return;
    const blob = new Blob([report.report_text], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `hallucination-report-q${questionId}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  }

  function downloadJson() {
    if (!report?.report_json) return;
    const blob = new Blob([JSON.stringify(report.report_json, null, 2)], {
      type: 'application/json',
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `hallucination-report-q${questionId}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <section className="bg-white rounded-xl border border-gray-200 shadow-sm p-5">
      <div className="flex flex-wrap items-center justify-between gap-2 mb-4">
        <h2 className="text-lg font-semibold text-gray-800">Verification Report</h2>
        <div className="flex gap-2 flex-wrap">
          {!report && (
            <button
              onClick={fetchReport}
              disabled={loading}
              className="px-4 py-1.5 text-sm bg-blue-600 text-white rounded-lg
                         hover:bg-blue-700 disabled:opacity-50 transition-colors"
            >
              {loading ? 'Loading…' : 'Load Report'}
            </button>
          )}
          {report && (
            <>
              <button
                onClick={downloadText}
                className="px-3 py-1.5 text-sm bg-gray-100 text-gray-700 rounded-lg
                           hover:bg-gray-200 transition-colors"
              >
                ↓ Download .txt
              </button>
              <button
                onClick={downloadJson}
                className="px-3 py-1.5 text-sm bg-gray-100 text-gray-700 rounded-lg
                           hover:bg-gray-200 transition-colors"
              >
                ↓ Download .json
              </button>
              <button
                onClick={() => setShowJson((v) => !v)}
                className="px-3 py-1.5 text-sm bg-gray-100 text-gray-700 rounded-lg
                           hover:bg-gray-200 transition-colors"
              >
                {showJson ? 'Hide JSON' : 'View JSON'}
              </button>
            </>
          )}
        </div>
      </div>

      {error && (
        <p className="text-sm text-red-600 bg-red-50 rounded-lg px-3 py-2">{error}</p>
      )}

      {report && (
        <>
          <pre className="text-xs bg-gray-50 rounded-lg border border-gray-100 p-4
                          overflow-x-auto whitespace-pre-wrap leading-relaxed text-gray-700">
            {report.report_text}
          </pre>

          {showJson && (
            <pre className="mt-4 text-xs bg-gray-900 text-green-300 rounded-lg p-4
                            overflow-x-auto leading-relaxed">
              {JSON.stringify(report.report_json, null, 2)}
            </pre>
          )}
        </>
      )}
    </section>
  );
}
