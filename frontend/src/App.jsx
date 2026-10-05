import { useState, useCallback } from 'react';
import { submitQuestion, getQuestion, getQuestionClaims } from './api';
import QuestionForm from './components/QuestionForm';
import VerificationSummary from './components/VerificationSummary';
import AnswerPanel from './components/AnswerPanel';
import ClaimsList from './components/ClaimsList';
import ReportSection from './components/ReportSection';
import HistoryPanel from './components/HistoryPanel';
import LoadingSpinner from './components/LoadingSpinner';

export default function App() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null); // QuestionCreateResponse
  const [historyRefresh, setHistoryRefresh] = useState(0);

  // Load a previous question from history
  async function loadFromHistory(questionId) {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const [detail, claims] = await Promise.all([
        getQuestion(questionId),
        getQuestionClaims(questionId),
      ]);
      // Stitch into the same shape as QuestionCreateResponse
      setResult({
        question_id: detail.question_id,
        question: detail.question,
        answer: detail.answer,
        corrected_answer: detail.corrected_answer,
        overall_verdict: detail.overall_verdict,
        overall_confidence: detail.overall_confidence,
        claims,
        status: 'completed',
      });
    } catch (err) {
      setError(
        err?.response?.data?.detail ?? err.message ?? 'Failed to load question.'
      );
    } finally {
      setLoading(false);
    }
  }

  const handleSubmit = useCallback(async (question) => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await submitQuestion(question);
      setResult(data);
      setHistoryRefresh((n) => n + 1); // trigger history reload
    } catch (err) {
      const detail = err?.response?.data?.detail;
      setError(
        detail
          ? typeof detail === 'string'
            ? detail
            : JSON.stringify(detail)
          : err.message ?? 'An unexpected error occurred.'
      );
    } finally {
      setLoading(false);
    }
  }, []);

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 shadow-sm sticky top-0 z-10">
        <div className="max-w-6xl mx-auto px-4 py-3 flex items-center gap-3">
          <span className="text-xl" aria-hidden="true">🔍</span>
          <div>
            <h1 className="text-base font-bold text-gray-900 leading-none">
              Hallucination Detection &amp; Answer Verification
            </h1>
            <p className="text-xs text-gray-500 mt-0.5">
              Hybrid multi-source fact verification using Wikipedia, Wikidata &amp; DuckDuckGo
            </p>
          </div>
        </div>
      </header>

      <div className="max-w-6xl mx-auto px-4 py-6 grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Left sidebar: question form + history */}
        <aside className="lg:col-span-1 space-y-4">
          <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-4">
            <QuestionForm onSubmit={handleSubmit} loading={loading} />
          </div>

          <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-4">
            <h3 className="text-sm font-semibold text-gray-700 mb-2">Recent Questions</h3>
            <HistoryPanel onSelect={loadFromHistory} refreshSignal={historyRefresh} />
          </div>
        </aside>

        {/* Main content area */}
        <main className="lg:col-span-3 space-y-5">
          {/* Error */}
          {error && (
            <div
              role="alert"
              className="bg-red-50 border border-red-200 rounded-xl px-4 py-3"
            >
              <p className="text-sm font-medium text-red-700">Error</p>
              <p className="text-sm text-red-600 mt-0.5">{error}</p>
            </div>
          )}

          {/* Loading */}
          {loading && <LoadingSpinner />}

          {/* Results */}
          {!loading && result && (
            <>
              {/* Question display */}
              <div className="bg-blue-50 border border-blue-100 rounded-xl px-4 py-3">
                <p className="text-xs text-blue-500 uppercase font-semibold tracking-wide">
                  Question
                </p>
                <p className="text-sm text-blue-900 font-medium mt-0.5">
                  {result.question}
                </p>
              </div>

              {/* Summary */}
              <VerificationSummary result={result} />

              {/* Original + Corrected answers */}
              <AnswerPanel
                answer={result.answer}
                correctedAnswer={result.corrected_answer}
              />

              {/* Claims */}
              <ClaimsList claims={result.claims} />

              {/* Report */}
              <ReportSection questionId={result.question_id} />
            </>
          )}

          {/* Empty state */}
          {!loading && !result && !error && (
            <div className="flex flex-col items-center justify-center py-20 text-center">
              <span className="text-5xl mb-4" aria-hidden="true">🧠</span>
              <p className="text-gray-500 text-sm max-w-xs">
                Enter a question on the left to start the fact-verification pipeline.
              </p>
              <p className="text-gray-400 text-xs mt-2 max-w-xs">
                The system will generate an LLM answer, extract claims, and verify
                each one against Wikipedia, Wikidata, and DuckDuckGo.
              </p>
            </div>
          )}
        </main>
      </div>

      {/* Footer */}
      <footer className="mt-12 border-t border-gray-200 bg-white py-4">
        <p className="text-center text-xs text-gray-400">
          Confidence scores are heuristic indicators, not calibrated probabilities.
          Multi-source agreement is supporting evidence, not proof of correctness.
        </p>
      </footer>
    </div>
  );
}
