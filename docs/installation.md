# Installation

### 1. Get the source

```bash
git clone https://github.com/System64fumo/kappai
cd kappai
```

### 2. Install dependencies

You need a C compiler, `make`, `pkg-config`, json-c, libmicrohttpd and ncurses.<br>
The GPU backends are optional and need the extras at the end of each line, so drop those if you only want the CPU.

**Debian / Ubuntu**
```bash
sudo apt install build-essential pkg-config libjson-c-dev libmicrohttpd-dev libncurses-dev libvulkan-dev glslc libegl-dev libgles-dev libgbm-dev
```

**Arch**
```bash
sudo pacman -S base-devel pkgconf json-c libmicrohttpd ncurses vulkan-headers vulkan-icd-loader shaderc mesa libglvnd
```

**Fedora**
```bash
sudo dnf install gcc make pkgconf-pkg-config json-c-devel libmicrohttpd-devel ncurses-devel vulkan-loader-devel glslc mesa-libEGL-devel mesa-libGLES-devel mesa-libgbm-devel
```

**macOS** (CPU only)
```bash
brew install pkg-config json-c libmicrohttpd ncurses
```

Package names shift between releases. If something is missing, `make config` or `make` will tell you.

### 3. Configure

```bash
make config BUILD=release
```

Add `BACKENDS=vulkan`, `BACKENDS=opengl` or `BACKENDS=vulkan,opengl` to build the GPU backends too.

`BUILD=release` is the one you want for actual use. The default is a debug build with sanitizers, which is great for development and painfully slow for inference.

### 4. Build

```bash
make -j$(nproc)
```

That builds `kappai`, `kappai-cli`, `kappai-server` and `kappai-diffusion` into `build/`. Two more are available if you want them:

```bash
make kappai-test       # backend regression tests
make kappai-monitor    # live monitor viewer
```

Now head over to [usage.md](usage.md) to run something.
