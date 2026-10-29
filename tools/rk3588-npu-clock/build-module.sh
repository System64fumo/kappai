#!/bin/sh
set -eu

if [ "$#" -ne 1 ]; then
	printf 'Usage: %s <linux-source-tree>\n' "$0" >&2
	exit 2
fi

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
source_root=$(CDPATH= cd -- "$1" && pwd)
kernel_release=$(uname -r)
expected_release=7.0.0-38-generic
kernel_build=/lib/modules/$kernel_release/build
output_dir=$repo_root/build/rocket-npu-clock

if [ "$kernel_release" != "$expected_release" ]; then
	printf 'Validated for %s; running kernel is %s.\n' "$expected_release" "$kernel_release" >&2
	exit 1
fi

if [ ! -d "$kernel_build" ]; then
	printf 'Missing kernel build directory: %s\n' "$kernel_build" >&2
	exit 1
fi

if [ ! -f "$source_root/drivers/accel/rocket/rocket_core.c" ]; then
	printf 'No mainline rocket driver under %s\n' "$source_root" >&2
	exit 1
fi

mkdir -p "$output_dir"
work_dir=$(mktemp -d "$output_dir/source.XXXXXX")
trap 'rm -rf "$work_dir"' EXIT HUP INT TERM
driver_source=$source_root/drivers/accel/rocket
driver_dir=$work_dir/drivers/accel/rocket
mkdir -p "$driver_dir"
for file in "$driver_source"/*.c; do
	case "$file" in
		*.mod.c) continue ;;
	esac
	cp "$file" "$driver_dir/"
done
cp "$driver_source"/*.h "$driver_source/Makefile" "$driver_dir/"
patch --batch --fuzz=0 -p1 -d "$work_dir" < "$repo_root/tools/rk3588-npu-clock/rocket-rk3588-7.0.0.patch"

cat > "$driver_dir/Makefile" <<'EOF'
obj-m := rocket.o
rocket-y := rocket_core.o rocket_device.o rocket_drv.o rocket_gem.o rocket_job.o
EOF

make -C "$kernel_build" M="$driver_dir" W=1 modules
module=$driver_dir/rocket.ko
vermagic=$(modinfo -F vermagic "$module")
srcversion=$(modinfo -F srcversion "$module")

case "$vermagic" in
	"$kernel_release"*) ;;
	*) printf 'Built vermagic does not match %s: %s\n' "$kernel_release" "$vermagic" >&2; exit 1 ;;
esac

if [ "$srcversion" != DE30E6454688A213AC43B11 ]; then
	printf 'Unexpected module srcversion: %s\n' "$srcversion" >&2
	exit 1
fi

install -D -m 0644 "$module" "$output_dir/rocket.ko"
printf 'Built %s with srcversion %s\n' "$output_dir/rocket.ko" "$srcversion"
