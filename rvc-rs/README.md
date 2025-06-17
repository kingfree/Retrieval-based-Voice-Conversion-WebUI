# rvc-rs Workspace

This directory contains a Rust rewrite of the project structured as a
Cargo workspace:

- `rvc-lib` – core library crate
- `rvc-tauri` – Tauri-based desktop application
- `ui` – Vue front-end built with Vite

## Building

1. Install Rust.
2. Build the library:
   ```bash
   cargo check -p rvc-lib
   ```
3. Install Node.js and run the front-end build:
   ```bash
   cd ui
   npm install
   npm run build
   ```
4. Build the Tauri application (optional, may fail if system
   dependencies like `glib` are missing):
   ```bash
   cargo tauri dev -p rvc-tauri
   ```
