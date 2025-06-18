# AGENTS

## Project Vision
The goal of this repository is to rewrite the original Retrieval-based Voice Conversion WebUI in Rust. The new Rust workspace (`rvc-rs`) contains:

- `rvc-lib`: core functionality as a Rust library, use `tch`
- `ui`: a Vue 3 front-end built with Vite, bundled into the Tauri app (`ui/src-tauri`)

The project aims to achieve feature parity with `gui_v1.py` while providing better performance and maintainability through Rust and Tauri.

## Architecture Principles

### Separation of Concerns
- **`rvc-lib`**: Contains ALL core functionality including:
  - Model loading and management
  - Audio processing pipelines
  - Parameter loading and saving
  - Configuration management
  - Real-time voice conversion logic
  - Device management
  - All business logic and data processing

- **Tauri (`ui/src-tauri`)**: Acts ONLY as a bridge between frontend and backend:
  - Receives requests from the Vue frontend
  - Calls corresponding methods from `rvc-lib`
  - Returns results to the frontend
  - Pushes events/progress updates to the frontend
  - Contains NO actual functionality or business logic
  - Should NOT duplicate any logic that exists in `rvc-lib`

### Key Rules
1. **No Logic Duplication**: If functionality exists in `rvc-lib`, Tauri must NOT re-implement it
2. **Tauri as Mediator**: Tauri functions should be thin wrappers that delegate to `rvc-lib`
3. **State Management**: All application state should be managed in `rvc-lib`, not in Tauri
4. **Parameter Handling**: All parameter validation, loading, and saving should happen in `rvc-lib`

## Guidelines
- Always run `cargo check -p rvc-lib --manifest-path rvc-rs/Cargo.toml` after modifying Rust code.
- Always run `cargo check --manifest-path rvc-rs/ui/src-tauri/Cargo.toml` after modifying Tauri code.
- Always run `npm run build` in `rvc-rs/ui` after modifying the Vue front-end. Run `npm install` when dependencies change.
- Running the Tauri app itself is optional as it requires system dependencies like `glib`.
- When adding new functionality, implement it in `rvc-lib` first, then create a thin Tauri wrapper.
- Avoid naming conflicts between `rvc-lib` functions and Tauri command functions.