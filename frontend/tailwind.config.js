/**
 * Tailwind config for the Amphion demo frontend.
 *
 * The frontend is plain HTML + vanilla JS (no bundler), so Tailwind is
 * pre-compiled once via ``scripts/build_tailwind.sh`` instead of being
 * loaded from ``cdn.tailwindcss.com`` on every navigation. The content
 * globs below feed the JIT scanner; both static HTML and the JS files
 * that build markup with template strings need to be covered, otherwise
 * dynamically-injected classes (e.g. the sidebar's ``app-nav-item`` or
 * emotion-app's history list) get tree-shaken away.
 *
 * Whenever a new Tailwind utility class is introduced anywhere in
 * ``frontend/*.html`` or ``frontend/*.js`` the build script must be
 * re-run so ``frontend/tailwind.css`` picks it up.
 */
module.exports = {
  content: [
    './*.html',
    './*.js',
  ],
  theme: {
    extend: {},
  },
  plugins: [],
};
