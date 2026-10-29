# RK3588 mainline NPU clock module

This directory contains the tested, temporary module patch and tools for the
mainline Linux `rocket` driver. It changes the active compute clock from the
device tree's 200 MHz setting to 600 MHz at runtime, then parks it back at
200 MHz when all three NPU cores suspend. The voltage patch keeps the NPU rail
at the tested 800 mV floor. The switch tool does not edit the device tree,
install a module, or change boot configuration.

The patch includes the clock and voltage controls, clock-name lookup, the
buffer lifetime fix, and the IOMMU/reset/IRQ fixes used by the tested driver.
The IOMMU changes are backported to the Ubuntu kernel source version listed
below; this patch is not a general Linux kernel patch.

## Validated configuration

- RK3588 with the mainline `rocket` accelerator driver
- Running kernel `7.0.0-38-generic`, Ubuntu source package `7.0.0-38.38`
- Candidate module srcversion `DE30E6454688A213AC43B11`
- Active NPU clock 600 MHz, idle clock 200 MHz, NPU rail 800 mV
- Three cores confirmed at 600 MHz and returned to runtime-suspended state

600 MHz is the highest frequency validated by this tool. The switch script
sets that fixed rate; it does not offer a higher-rate option. Do not change the
clock or voltage constants to try 1 GHz: higher settings have not been
validated here and previous driver experiments caused a board hang. Keep the
device tree at its stock 200 MHz rate.

The module is unsigned and loaded out of tree, so the kernel will report its
usual module-verification warning and taint flag. It is only loaded for the
current boot. Rebooting restores the distribution module.

## Build

Install matching headers, the Ubuntu Linux 7.0 source package, and build tools:

```sh
sudo apt install linux-source-7.0.0 linux-headers-$(uname -r) build-essential kmod acl
```

Extract the source package, then build the patched module against the running
kernel's headers and symbol versions:

```sh
mkdir -p build/linux-source-7.0.0
tar -xjf /usr/src/linux-source-7.0.0/linux-source-7.0.0.tar.bz2 \
  -C build/linux-source-7.0.0 --strip-components=1
tools/rk3588-npu-clock/build-module.sh build/linux-source-7.0.0
```

The script requires the validated kernel release, applies the patch to a
temporary copy of `drivers/accel/rocket/`, builds with `W=1`, and checks the
module's vermagic and srcversion. It writes the resulting `rocket.ko` under
the ignored `build/rocket-npu-clock/` directory. No binary module is stored in
Git.

## Load and restore

Close all applications using the NPU and verify the preflight checks:

```sh
python3 tools/rk3588-npu-clock/switch-module.py check
```

Load the candidate for this boot only:

```sh
sudo python3 tools/rk3588-npu-clock/switch-module.py load
```

The tool checks that the driver is idle, power management is automatic, the
device-tree rate is 200 MHz, and the NPU rail is 800 mV. It checks the module
identity before replacing the driver, then verifies all three per-core clock
and voltage messages and the 200 MHz idle park. When invoked with `sudo`, it
restores the invoking user's read/write access to `/dev/accel/accel0`.
The load and check actions need `/sys/kernel/debug/clk/clk_summary`; restore
can still reload the stock module if that clock debugfs file is unavailable.

Return to the distribution driver with:

```sh
sudo python3 tools/rk3588-npu-clock/switch-module.py restore
```

If the NPU is busy, any preflight check fails, or post-load verification does
not match, stop using the NPU and inspect the kernel log. A reboot loads the
distribution driver because the candidate is never installed.

## Patch source

`rocket-rk3588-7.0.0.patch` is the combined driver-source change used to build
the validated module. Its clock and voltage changes follow the
[RK3588 NPU clock notes](https://github.com/gregordinary/rockchip-npu-notes/blob/main/perf/clock.md)
and the patches at revision
[`ac458850`](https://github.com/gregordinary/patches/tree/ac458850f0c4f522b47a12f73c9ce101867d5369/rocket).
The IOMMU/reset backport is included because clock changes alone were not the
driver version used for the final NPU runs.

The kernel-driver patch is GPL-2.0-only; see [LICENSE](LICENSE). The build and
switch scripts are part of kappai and follow the repository license.
