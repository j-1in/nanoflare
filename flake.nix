{
  description = "Rust + CUDA Development Environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;     # Required for NVIDIA/CUDA
            cudaSupport = true;     # Build packages with CUDA support
          };
        };

        # READ RUST TOOLCHAIN; ensure rust-toolchain.toml exists in the same dir.
        rustToolchain = (builtins.fromTOML (builtins.readFile ./rust-toolchain.toml));
        rustcVersion = rustToolchain.toolchain.channel;

        # CUDA PACKAGES
        cudaPkg = pkgs.cudaPackages;
      in
      {
        devShells.default = pkgs.mkShell {
          name = "rust-cuda-env";
          strictDeps = true;

          # --- BUILD INPUTS ---
          # Tools needed at compile-time (compilers, build systems)
          nativeBuildInputs = with pkgs; [
            # Rust Tools
            rustup
            pkg-config
            rustPlatform.bindgenHook # Crucial for FFI

            # C/C++ Tools (Needed for compiling the C side of FFI)
            cmake
            ninja
            gnumake

            # Helper to setup CUDA environment variables automatically
            cudaPkg.cuda_nvcc
          ];

          # Libraries needed at runtime or for linking
          buildInputs = with pkgs; [
            openssl
            cudaPkg.cudatoolkit
            linuxPackages.nvidia_x11 # Drivers/Libs for dynamic linking
          ];

          # --- ENVIRONMENT VARIABLES ---

          # Rust version from your toml file
          RUSTC_VERSION = rustcVersion;

          # Point tools to the CUDA installation
          CUDA_PATH = "${cudaPkg.cudatoolkit}";

          # Bindgen needs to know where libclang is to parse C headers
          LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";

          # Help bindgen find CUDA headers (essential for FFI!)
          BINDGEN_EXTRA_CLANG_ARGS = "-I${cudaPkg.cudatoolkit}/include -I${pkgs.llvmPackages.libclang.lib}/lib/clang/${pkgs.llvmPackages.clang.version}/include";

          # Ensure the dynamic linker can find the NVIDIA drivers and CUDA libraries
          LD_LIBRARY_PATH = "${pkgs.lib.makeLibraryPath [
            pkgs.linuxPackages.nvidia_x11
            cudaPkg.cudatoolkit
            pkgs.openssl
          ]}";

          # --- SHELL HOOK ---
          shellHook = ''
            # 1. Setup Rust Path
            export PATH="${"\${CARGO_HOME:-~/.cargo}/bin"}:$PATH"

            # 2. Generate .clangd for IDE support (C++ / CUDA side)
            cat > .clangd <<EOF
            CompileFlags:
              Add:
                - "--cuda-path=${cudaPkg.cudatoolkit}"
                - "-ferror-limit=0"
                - "-x"
                - "cuda"
                # Updated to Ampere/Ada (RTX 3000/4000). Adjust if needed.
                - "--cuda-gpu-arch=sm_86"
                - "-I${cudaPkg.cudatoolkit}/include"
              Remove:
                # strip CUDA fatbin args
                - "-Xfatbin*"
                # strip CUDA arch flags
                - "-arch=native"
                - "-gencode*"
                - "--generate-code*"
                # strip CUDA flags unknown to clang
                - "-ccbin*"
                - "--compiler-options*"
                - "--expt-extended-lambda"
                - "--expt-relaxed-constexpr"
                - "-forward-unknown-to-host-compiler"
                - "-Werror=cross-execution-space-call"
            EOF

            echo "🚀 Rust + CUDA Environment Loaded"
            echo "🦀 Rust Channel: $RUSTC_VERSION"
            echo "🔌 CUDA Path: $CUDA_PATH"
          '';
        };
      }
    );
}
