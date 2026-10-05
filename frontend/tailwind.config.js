/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        verdict: {
          verified: '#16a34a',
          partial: '#ca8a04',
          unsupported: '#6b7280',
          contradicted: '#dc2626',
        },
      },
    },
  },
  plugins: [],
}

