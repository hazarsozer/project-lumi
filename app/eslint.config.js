import js from "@eslint/js";
import tseslint from "typescript-eslint";
import reactHooks from "eslint-plugin-react-hooks";
import reactRefresh from "eslint-plugin-react-refresh";

export default tseslint.config(
  // Base recommended JS rules
  js.configs.recommended,
  // TypeScript-eslint recommended rules
  ...tseslint.configs.recommended,
  {
    // Apply to all TS/TSX source files
    files: ["src/**/*.{ts,tsx}"],
    plugins: {
      "react-hooks": reactHooks,
      "react-refresh": reactRefresh,
    },
    rules: {
      // React Hooks — core correctness rules only (hooks v7 recommended has
      // many experimental rules not yet stable; we pin just the two proven ones)
      "react-hooks/rules-of-hooks": "error",
      "react-hooks/exhaustive-deps": "warn",

      // react-refresh: disabled — several files intentionally export a mix of
      // components + constants/types from the same module (e.g. SettingsTabs.tsx
      // exports DEFAULT_SETTINGS alongside tab components by design).
      // This is an HMR granularity hint, not a correctness rule.
      "react-refresh/only-export-components": "off",

      // TypeScript tuning: the codebase uses deliberate `as` casts in a few
      // places (IPC wire-type narrowing); surface as warn, not error.
      "@typescript-eslint/no-explicit-any": "warn",

      // console.warn/error is used in the IPC client for intentional diagnostics;
      // loosen to warn so existing calls don't block CI.
      "no-console": ["warn", { allow: ["warn", "error"] }],

      // Unused vars: permit leading-underscore convention for intentionally
      // unused params (e.g. mock client _event).
      "@typescript-eslint/no-unused-vars": [
        "error",
        { argsIgnorePattern: "^_", varsIgnorePattern: "^_" },
      ],
    },
  },
  {
    // Ignore build artifacts, Tauri Rust dir, Vite/Vitest config files
    ignores: [
      "dist/**",
      "src-tauri/**",
      "*.config.{js,ts}",
      "src/vite-env.d.ts",
    ],
  },
);
