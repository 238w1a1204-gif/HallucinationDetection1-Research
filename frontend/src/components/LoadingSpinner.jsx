/**
 * Full-width loading indicator shown while the pipeline runs.
 */
export default function LoadingSpinner({ message = 'Running verification pipeline…' }) {
  return (
    <div className="flex flex-col items-center justify-center py-16 gap-4">
      <div
        className="w-12 h-12 rounded-full border-4 border-gray-200 border-t-blue-600 animate-spin"
        role="status"
        aria-label="Loading"
      />
      <p className="text-sm text-gray-500">{message}</p>
      <p className="text-xs text-gray-400 max-w-xs text-center">
        First run may take 1–2 minutes while the LLM model loads. Subsequent
        requests are faster.
      </p>
    </div>
  );
}
