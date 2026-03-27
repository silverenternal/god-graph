fn main() {
    // Declare custom cfg for nightly Rust detection
    println!("cargo::rustc-check-cfg=cfg(nightly)");
}
