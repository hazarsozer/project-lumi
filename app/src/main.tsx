import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import { GLOBAL_KEYFRAMES } from "./styles/tokens";

const _ks = document.createElement("style");
_ks.textContent = GLOBAL_KEYFRAMES;
document.head.appendChild(_ks);

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
