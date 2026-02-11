#[cfg(target_arch = "wasm32")]
use leptos::mount::mount_to_body;
#[cfg(target_arch = "wasm32")]
use nghe_frontend::Body;

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    eprintln!("`nghe_frontend` is a WebAssembly (wasm32) app and can't be run as a native binary.");
    eprintln!();
    eprintln!("Run it with Trunk instead:");
    eprintln!("  cd nghe-frontend");
    eprintln!("  trunk serve");
    std::process::exit(1);
}

#[cfg(target_arch = "wasm32")]
fn main() {
    console_error_panic_hook::set_once();
    mount_to_body(Body);
}
