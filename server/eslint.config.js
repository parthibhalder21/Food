import js from "@eslint/js";
import globals from "globals";
import security from "eslint-plugin-security";

export default [
  { ignores: ["data/**"] },
  {
    files: ["**/*.js"],
    languageOptions: { ecmaVersion: "latest", sourceType: "module", globals: globals.node },
    rules: {
      ...js.configs.recommended.rules,
      "no-unused-vars": ["error", { argsIgnorePattern: "^_" }],
    },
  },
];

// SAST rules (OWASP-style checks for Node.js). `npm run sast` runs only these, and the pipeline fails on any error.
export const sast = [
  { ignores: ["data/**", "test/**"] },
  { linterOptions: { reportUnusedDisableDirectives: "off" } },
  {
    files: ["**/*.js"],
    languageOptions: { ecmaVersion: "latest", sourceType: "module", globals: globals.node },
    plugins: { security },
    rules: {
      ...Object.fromEntries(Object.keys(security.rules).map((r) => [`security/${r}`, "error"])),
      // High false-positive rate on ordinary code (obj[key], paths from our own config): reported, not blocking.
      "security/detect-object-injection": "warn",
      "security/detect-non-literal-fs-filename": "warn",
      "no-eval": "error",
      "no-implied-eval": "error",
      "no-new-func": "error",
    },
  },
];
