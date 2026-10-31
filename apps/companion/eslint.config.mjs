import js from '@eslint/js'
import tseslint from 'typescript-eslint'

export default tseslint.config(
  { ignores: ['node_modules/**', 'dist/**', 'out/**'] },
  js.configs.recommended,
  ...tseslint.configs.recommended,
  {
    files: ['**/*.ts'],
    rules: {
      '@typescript-eslint/no-unused-vars': ['error', { argsIgnorePattern: '^_', varsIgnorePattern: '^_' }],
    },
  },
  {
    // The app's pages run in its sandboxed renderer (a browser), as plain scripts.
    files: ['src/app/ui/**/*.js'],
    languageOptions: { sourceType: 'script', globals: { window: 'readonly', document: 'readonly', location: 'readonly', URL: 'readonly', DOMException: 'readonly', console: 'readonly', setTimeout: 'readonly', setInterval: 'readonly', clearTimeout: 'readonly', performance: 'readonly', navigator: 'readonly', localStorage: 'readonly' } },
  },
  {
    // The executor is the final authority on this computer: it must not reach the harness, the app or the network
    // stack directly; the harness and the app call it, never the other way round.
    files: ['src/executor/**/*.ts'],
    rules: {
      'no-restricted-imports': ['error', { patterns: [{ group: ['../harness/*', '../app/*', 'electron'], message: 'The executor does not depend on the harness or the app.' }] }],
    },
  },
)
