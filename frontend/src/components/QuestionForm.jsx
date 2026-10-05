import { useState } from 'react';

/**
 * Text input + submit button for entering a question.
 * Calls `onSubmit(questionText)` when submitted.
 */
export default function QuestionForm({ onSubmit, loading }) {
  const [question, setQuestion] = useState('');

  function handleSubmit(e) {
    e.preventDefault();
    const trimmed = question.trim();
    if (!trimmed) return;
    onSubmit(trimmed);
  }

  return (
    <form onSubmit={handleSubmit} className="w-full">
      <label
        htmlFor="question-input"
        className="block text-sm font-medium text-gray-700 mb-1"
      >
        Enter a question to fact-check
      </label>
      <div className="flex gap-2">
        <textarea
          id="question-input"
          rows={3}
          className="flex-1 rounded-lg border border-gray-300 px-4 py-2 text-sm
                     focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none
                     disabled:opacity-50"
          placeholder="e.g. Who was Albert Einstein and what is he known for?"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          disabled={loading}
          aria-label="Question input"
        />
        <button
          type="submit"
          disabled={loading || !question.trim()}
          className="self-end px-5 py-2 rounded-lg bg-blue-600 text-white text-sm font-medium
                     hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500
                     disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {loading ? 'Checking…' : 'Verify'}
        </button>
      </div>
    </form>
  );
}
