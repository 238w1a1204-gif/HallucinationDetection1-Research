/**
 * Centralised API client.
 * All URLs are derived from VITE_API_BASE_URL so nothing is hard-coded
 * throughout the component tree.
 */
import axios from 'axios';

const BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

const client = axios.create({
  baseURL: `${BASE_URL}/api`,
  headers: { 'Content-Type': 'application/json' },
  timeout: 300_000, // 5 min — pipeline can be slow on first run (model download)
});

/** GET /api/health */
export async function getHealth() {
  const { data } = await client.get('/health');
  return data;
}

/**
 * POST /api/questions
 * @param {string} question
 * @returns {Promise<object>} full QuestionCreateResponse
 */
export async function submitQuestion(question) {
  const { data } = await client.post('/questions', { question });
  return data;
}

/**
 * GET /api/questions
 * @param {{ skip?: number, limit?: number }} params
 */
export async function listQuestions(params = {}) {
  const { data } = await client.get('/questions', { params });
  return data;
}

/**
 * GET /api/questions/:id
 * @param {number} id
 */
export async function getQuestion(id) {
  const { data } = await client.get(`/questions/${id}`);
  return data;
}

/**
 * GET /api/questions/:id/claims
 * @param {number} id
 */
export async function getQuestionClaims(id) {
  const { data } = await client.get(`/questions/${id}/claims`);
  return data;
}

/**
 * GET /api/questions/:id/report
 * @param {number} id
 */
export async function getReport(id) {
  const { data } = await client.get(`/questions/${id}/report`);
  return data;
}
