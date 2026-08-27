fn main() {
    // `torch-sys` links LibTorch's CUDA backend, but GNU ld can discard it
    // under `--as-needed` because CUDA operators are registered through static
    // initializers rather than directly referenced symbols. Retain the backend
    // so `tch::Cuda::is_available()` and CUDA tensor dispatch work at runtime.
    println!("cargo:rustc-link-arg=-Wl,--push-state,--no-as-needed,-ltorch_cuda,--pop-state");
}
