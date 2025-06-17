# rvc-tauri

This crate provides the Tauri-based desktop application for `rvc-rs`.
It depends on the `rvc-lib` crate for core functionality and loads the
Vue interface from `../ui/dist`.

## Building

The `rvc-tauri` crate is part of the `rvc-rs` workspace. After building
the front-end and ensuring Rust and the Tauri CLI are installed, run:

```bash
cargo tauri dev -p rvc-tauri
```

Note: building may fail in environments missing system libraries (e.g.,
`glib`) required by Tauri.
