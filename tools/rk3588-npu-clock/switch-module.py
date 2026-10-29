#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import struct
import subprocess
import time

EXPECTED_KERNEL = "7.0.0-38-generic"
STOCK_SRCVERSION = "4AF8378A5F1A9C87104B828"
CLOCK_ONLY_SRCVERSION = "1EEAC9ADA1C13D157BFFBCE"
CANDIDATE_SRCVERSION = "DE30E6454688A213AC43B11"
NPU_ADDRESSES = ("fdab0000", "fdac0000", "fdad0000")
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODULE = ROOT / "build/rocket-npu-clock/rocket.ko"


def read_srcversion(module="rocket"):
	return subprocess.check_output(["modinfo", "-F", "srcversion", str(module)], text=True).strip()


def check_system(action, module):
	if os.uname().release != EXPECTED_KERNEL:
		raise SystemExit(f"Validated for {EXPECTED_KERNEL}; running kernel is {os.uname().release}.")

	loaded = Path("/sys/module/rocket/srcversion")
	if not loaded.exists():
		raise SystemExit("The mainline rocket module is not loaded.")
	current = loaded.read_text().strip()
	if current not in (STOCK_SRCVERSION, CLOCK_ONLY_SRCVERSION, CANDIDATE_SRCVERSION):
		raise SystemExit(f"Unrecognized rocket module srcversion {current}; leaving it untouched.")
	if Path("/sys/module/rocket/refcnt").read_text().strip() != "0":
		raise SystemExit("NPU clients are still open; close them before switching the module.")

	devices = [Path("/sys/devices/platform") / f"{address}.npu" for address in NPU_ADDRESSES]
	for device in devices:
		if (device / "power/control").read_text().strip() != "auto":
			raise SystemExit(f"{device.name} is pinned on; power/control must be auto.")
		if (device / "power/runtime_status").read_text().strip() != "suspended":
			raise SystemExit(f"{device.name} is active; wait for it to suspend.")
		property_path = Path("/proc/device-tree") / f"npu@{device.name.removesuffix('.npu')}" / "assigned-clock-rates"
		if struct.unpack(">I", property_path.read_bytes())[0] != 200000000:
			raise SystemExit("The NPU boot clock is not the stock 200 MHz; restore the original DT first.")

	rails = [rail for rail in Path("/sys/class/regulator").glob("regulator.*")
		 if (rail / "name").read_text().strip() == "vdd_npu_s0"]
	if len(rails) != 1 or int((rails[0] / "microvolts").read_text()) != 800000:
		raise SystemExit("Expected the NPU rail vdd_npu_s0 at 800 mV.")
	clock_summary = Path("/sys/kernel/debug/clk/clk_summary")
	if not clock_summary.is_file() and action != "restore":
		raise SystemExit("Clock debugfs is unavailable; mount debugfs before switching modules.")
	if not clock_summary.is_file():
		clock_summary = None

	if read_srcversion("rocket") != STOCK_SRCVERSION:
		raise SystemExit("The installed modprobe rocket module is not the expected stock module.")
	if action != "restore":
		if not module.is_file():
			raise SystemExit(f"Missing candidate module: {module}")
		if subprocess.check_output(["modinfo", "-F", "vermagic", str(module)], text=True).split()[0] != EXPECTED_KERNEL:
			raise SystemExit("Candidate vermagic does not match the running kernel.")
		if read_srcversion(module) != CANDIDATE_SRCVERSION:
			raise SystemExit("Candidate module srcversion does not match the validated build.")

	print("Preflight passed: matching kernel, idle cores, auto power control, 200 MHz DT, 800 mV rail.")
	return current, devices, rails[0], clock_summary


def dmesg_lines():
	result = subprocess.run(["dmesg", "--color=never"], capture_output=True, text=True)
	return result.stdout.splitlines() if result.returncode == 0 else []


def set_device_access():
	user = os.environ.get("SUDO_USER")
	device = Path("/dev/accel/accel0")
	if not device.exists():
		raise SystemExit("The rocket device node was not created.")
	if user:
		subprocess.run(["setfacl", "-m", f"u:{user}:rw", str(device)], check=True)


def idle_clock_hz(clock_summary):
	if clock_summary is None:
		return None
	for line in clock_summary.read_text().splitlines():
		fields = line.split()
		if fields and fields[0] == "scmi_clk_npu":
			return int(fields[4])
	raise SystemExit("scmi_clk_npu is missing from clk_summary.")


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("action", choices=("check", "load", "restore"), nargs="?", default="check")
	parser.add_argument("--module", type=Path, default=DEFAULT_MODULE)
	args = parser.parse_args()
	if args.action != "check" and os.geteuid() != 0:
		raise SystemExit("Root access is required to replace the running kernel module.")
	current, devices, rail, clock_summary = check_system(args.action, args.module)
	if args.action == "check":
		return
	if args.action == "load" and current == CANDIDATE_SRCVERSION:
		raise SystemExit("The candidate is already loaded.")
	if args.action == "restore" and current == STOCK_SRCVERSION:
		raise SystemExit("The stock distribution module is already loaded.")

	before = dmesg_lines()
	print("Replacing the idle rocket module for this boot only.", flush=True)
	subprocess.run(["rmmod", "rocket"], check=True)
	if args.action == "load":
		result = subprocess.run(["insmod", str(args.module), "rocket_npu_clk_hz=600000000",
							 "rocket_autosuspend_ms=1000"])
		if result.returncode:
			subprocess.run(["modprobe", "rocket"], check=True)
			subprocess.run(["udevadm", "settle", "--timeout=10"], check=True)
			set_device_access()
			raise SystemExit("Candidate load failed; requested the stock driver again.")
	else:
		subprocess.run(["modprobe", "rocket"], check=True)

	subprocess.run(["udevadm", "settle", "--timeout=10"], check=True)
	set_device_access()
	time.sleep(2)
	after = dmesg_lines()
	new_log = after[len(before):] if after[:len(before)] == before else after[-80:]
	print("\n".join(new_log), flush=True)
	states = {device.name: (device / "power/runtime_status").read_text().strip()
			  for device in devices}
	actual = Path("/sys/module/rocket/srcversion").read_text().strip()
	clock_hz = idle_clock_hz(clock_summary)
	voltage_uv = int((rail / "microvolts").read_text())
	expected = CANDIDATE_SRCVERSION if args.action == "load" else STOCK_SRCVERSION
	if (actual != expected or (clock_summary and clock_hz != 200000000) or
			any(state != "suspended" for state in states.values())):
		raise SystemExit("Post-switch verification failed; stop NPU work and inspect dmesg.")
	if voltage_uv != 800000:
		raise SystemExit(f"Unexpected NPU rail voltage after switch: {voltage_uv} uV.")

	if args.action == "load":
		clock_cores = {int(line.split("core ")[1].split()[0]) for line in new_log
					   if "NPU clk -> 600000000 Hz (reads back 600000000)" in line}
		voltage_cores = {int(line.split("core ")[1].split()[0]) for line in new_log
						 if "NPU vdd -> 800000 uV (reads back 800000 uV)" in line}
		if clock_cores != {0, 1, 2} or voltage_cores != {0, 1, 2}:
			raise SystemExit("Per-core 600 MHz / 800 mV confirmations are missing; inspect dmesg.")
		print("Candidate loaded: three cores reached 600 MHz at 800 mV; idle clock parked at 200 MHz.")
	else:
		if clock_summary:
			print("Stock distribution module restored; all cores suspended at 200 MHz.")
		else:
			print("Stock distribution module restored; all cores suspended; idle clock was not readable.")


if __name__ == "__main__":
	main()
