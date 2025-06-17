# AGENTS

## Project Vision
The goal of this repository is to rewrite the original Retrieval-based Voice Conversion WebUI in Rust. The new Rust workspace (`rvc-rs`) contains:

- `rvc-lib`: core functionality as a Rust library
- `rvc-tauri`: a Tauri desktop application that uses `rvc-lib`
- `ui`: a Vue 3 front-end built with Vite, bundled into the Tauri app

The project aims to achieve feature parity with `gui_v1.py` while providing better performance and maintainability through Rust and Tauri.

## Guidelines
- Always run `cargo check -p rvc-lib --manifest-path rvc-rs/Cargo.toml` after modifying Rust code.
- Always run `npm run build` in `rvc-rs/ui` after modifying the Vue front-end. Run `npm install` when dependencies change.
- Running the Tauri app itself is optional as it requires system dependencies like `glib`.
- Do not commit `Cargo.lock` or `package-lock.json`.
