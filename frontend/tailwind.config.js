/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                darkBase: '#0b0f19',
                darkPanel: '#1a2235',
                accentReal: '#10b981',
                accentFake: '#ef4444',
            }
        },
    },
    plugins: [],
}