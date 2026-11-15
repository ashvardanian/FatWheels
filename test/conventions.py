#!/usr/bin/env python3
"""Turns every rule of FatWheels' two capability flags into PASS or FAIL.

    python3 test/conventions.py cpu        # in `fatwheels:cpu`, or x86-64 Linux with GCC 15, Clang 23 and SDE
    python3 test/conventions.py gpu        # in `fatwheels:gpu`: CUDA and HIP compiles, entries, instructions
    python3 test/conventions.py gpu --run  # on a machine with an NVIDIA GPU

Only the standard library is used, so it runs in every image as it is.
"""

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
failures = []


def report(passed: bool, rule: str, detail: str = "") -> None:
    print(f"{'PASS' if passed else 'FAIL'}  {rule}" + (f"  ({detail})" if detail else ""))
    if not passed:
        failures.append(rule)


def run(*command, check=True, **kwargs) -> subprocess.CompletedProcess:
    result = subprocess.run([str(part) for part in command], capture_output=True, text=True, **kwargs)
    if check and result.returncode != 0:
        sys.exit(f"{' '.join(map(str, command))} failed:\n{result.stdout}{result.stderr}")
    return result


def build(preset: str) -> Path:
    run("cmake", "--preset", preset, cwd=root)
    run("cmake", "--build", "--preset", preset, cwd=root)
    return root / f"build_{preset}"


def definitions(tree: Path, source: str) -> dict[str, str]:
    """The `-D` definitions CMake passes to the unit compiling `source`."""
    for entry in json.loads((tree / "compile_commands.json").read_text()):
        if entry["file"].endswith(source):
            return dict(re.findall(r"-D(\w+)=(\S+)", entry["command"]))
    return {}


def symbols(tree: Path, unit: str) -> set[str]:
    objects = list(tree.glob(f"CMakeFiles/fatwheels_static.dir/src/capabilities/{unit}.c.o"))
    if not objects:
        return set()
    return set(re.findall(r" T (fw_\w+)", run("nm", objects[0]).stdout))


def preprocess(compiler: str, flags: list[str], source: str) -> dict[str, str]:
    """The macros `source` sees under `flags`, with no build system around it."""
    result = run(compiler, *flags, "-I", root / "c", "-I", root / "include", "-dM", "-E", "-x", "c", "-", input=source)
    return dict(re.findall(r"#define (FATWHEELS_\w+) (\S+)", result.stdout))


def check_sources() -> None:
    types = (root / "c/fatwheels/types.h").read_text().splitlines()
    readers = [line for line in types if line.startswith("#define FATWHEELS_ARCH_") and "TARGET" in line]
    report(not readers, "no ARCH flag is defined from a TARGET", "; ".join(readers))


def check_cpu() -> None:
    clang = build("clang-floor")
    units = {unit: symbols(clang, unit) for unit in ("serial", "haswell", "skylake", "sapphireamx")}
    once = all(units[unit] == {f"fw_matmul_f32_{unit}"} for unit in units)
    report(once, "each capability unit defines its own kernel and no sibling's", str(units))
    dumps = {unit: run("objdump", "-d", next(clang.glob(f"CMakeFiles/fatwheels_static.dir/src/capabilities/{unit}.c.o"))).stdout for unit in units}
    report(bool(re.search(r"vfmadd\w+ps\s[^\n]*%ymm", dumps["haswell"])), "Haswell emits `vfmadd...ps ymm`")
    report(bool(re.search(r"vfmadd\w+ps\s[^\n]*%zmm", dumps["skylake"])), "Skylake emits `vfmadd...ps zmm`")
    report("tdpbf16ps" in dumps["sapphireamx"], "Sapphire AMX emits `tdpbf16ps`")

    floor = preprocess("clang-23", [], "#include <fatwheels/types.h>\n")
    raised = definitions(clang, "src/capabilities/haswell.c")
    report(floor.get("FATWHEELS_ARCH_X8664_HASWELL_") == "0" and raised.get("FATWHEELS_ARCH_X8664_HASWELL_") == "1",
           "at Clang's floor source sees no Haswell and the probe raises it")
    gcc_v3 = preprocess("gcc-15", ["-march=x86-64-v3"], "#include <fatwheels/types.h>\n")
    report(gcc_v3.get("FATWHEELS_ARCH_X8664_HASWELL_") == "1", "at x86-64-v3 source alone detects Haswell")

    dropped = build("clang-without-amx")
    flags = definitions(dropped, "src/fatwheels.c")
    report(flags.get("FATWHEELS_ARCH_X8664_SAPPHIREAMX_") == "1" and flags.get("FATWHEELS_TARGET_SAPPHIREAMX") == "0"
           and not (dropped / "CMakeFiles/fatwheels_static.dir/src/capabilities/sapphireamx.c.o").exists(),
           "a dropped target keeps its ARCH flag and loses its kernels")

    mono = build("gcc-monoversioned")
    flags = definitions(mono, "src/fatwheels.c")
    kept = {cap: flags.get(f"FATWHEELS_TARGET_{cap}") for cap in ("HASWELL", "SKYLAKE", "SAPPHIREAMX")}
    report(kept == {"HASWELL": "1", "SKYLAKE": "0", "SAPPHIREAMX": "0"} and flags.get("FATWHEELS_ARCH_X8664_SKYLAKE_") == "1",
           "a monoversioned build keeps what the floor implies, and ARCH still says what compiles", str(kept))
    result = run(mono / "fatwheels_test", check=False)
    report(result.returncode == 0 and "dispatched 0x2" in result.stdout,
           "a monoversioned build's tables hold one kernel per operation", result.stdout.strip().splitlines()[-1])

    # As the Skylake probe and unit see it: Skylake raised at a floor that enables neither.
    macros = preprocess("clang-23", ["-DFATWHEELS_ARCH_X8664_SKYLAKE_=1"], "#include <fatwheels/types.h>\n")
    report(macros.get("FATWHEELS_TARGET_SKYLAKE") == "1" and macros.get("FATWHEELS_TARGET_HASWELL") == "0"
           and macros.get("FATWHEELS_ARCH_X8664_HASWELL_") == "1",
           "raising Skylake opens Haswell's blocks but not Haswell's kernels")

    header_only = root / "test/header_only.c"
    plain = run("clang-23", "-O2", "-I", root / "c", "-I", root / "include", header_only, "-o", clang / "header_only")
    report(run(clang / "header_only", check=False).returncode == 0, "a header-only user builds and runs with no build system")
    vouched = run("clang-23", "-O2", "-DFATWHEELS_TARGET_SAPPHIREAMX=1", "-I", root / "c", "-I", root / "include",
                  header_only, "-o", clang / "header_only_vouched", check=False)
    report(vouched.returncode != 0 and "fw_matmul_f32_sapphireamx" in vouched.stderr,
           "a hand-set target without flags or a build fails to link, naming the kernel")
    del plain

    report(run(clang / "fatwheels_test_declarations", check=False).returncode == 0,
           "a unit whose flags enable nothing calls the library's Haswell kernel through its declaration")
    sde = shutil.which("sde64") or os.environ.get("SDE")
    runners = [("native", [])] + ([("SDE Skylake", [sde, "-skx", "--"]), ("SDE Sapphire Rapids", [sde, "-spr", "--"])] if sde else [])
    for name, prefix in runners:
        result = run(*prefix, clang / "fatwheels_test", check=False)
        report(result.returncode == 0, f"every runnable capability matches serial, {name}", result.stdout.strip().splitlines()[-1])
    if not sde:
        report(False, "SDE runs Skylake and Sapphire AMX", "sde64 not found")

    if shutil.which("qemu-aarch64") and (root / "cmake/toolchain-aarch64-gnu.cmake").exists():
        arm = build("aarch64-qemu")
        for cpu in ("max,sve=off,sme=off", "max,sme=off,sve-max-vq=1", "max,sve-max-vq=4"):
            result = run("qemu-aarch64", "-cpu", cpu, arm / "fatwheels_test", check=False)
            report(result.returncode == 0, f"AArch64 matches serial under QEMU {cpu}", result.stdout.strip().splitlines()[-1] if result.stdout else result.stderr[-200:])


def cubin_symbols(path: Path) -> tuple[set[str], set[str]]:
    """The kernel entries and `__device__` globals of every cubin in an object or fatbin."""
    entries, globals_ = set(), set()
    section = False
    for line in run("cuobjdump", "-elf", path).stdout.splitlines():
        if line.startswith(".section"):
            section = line.startswith(".section .symtab")
            continue
        fields = line.split()
        if section and len(fields) > 4:
            kind = fields[3]
            (entries if kind in ("0x2", "0x12") else globals_ if kind in ("0x1", "0x11") else set()).add(fields[-1])
    return entries, globals_


def check_cuda(execute: bool) -> None:
    cuda = build("cuda")
    objects = cuda / "CMakeFiles"
    units = {unit: cubin_symbols(next(objects.glob(f"fatwheels_{unit}_.dir/src/capabilities/{unit}.cu.o")))
             for unit in ("cuda", "ampere", "ada")}
    names = {unit: {entry for entry in entries if "fw_" in entry} for unit, (entries, _) in units.items()}
    report(all(any(f"fw_matmul_f32_{unit}_kernel_" in entry for entry in names[unit]) for unit in units)
           and not any("ampere" in entry or "ada" in entry for entry in names["cuda"])
           and not any("matmul_f32_ada" in entry for entry in names["ampere"]),
           "each generation unit holds its own kernels", str(names))
    report("fw_halves_ampere_" in units["ampere"][1] and not units["ada"][1],
           "Ampere's unit holds its table and Ada's unit does not", str({u: g for u, (_, g) in units.items()}))
    report(any("fw_zero_f32_ampere_kernel_" in entry for entry in names["ada"]),
           "a kernel shared by Ampere and Ada lives in both units")
    sass = {unit: run("cuobjdump", "-sass", next(objects.glob(f"fatwheels_{unit}_.dir/src/capabilities/{unit}.cu.o"))).stdout for unit in ("ampere", "ada")}
    report("HMMA" in sass["ampere"], "Ampere emits `HMMA` from `mma.sync` in bf16")
    report("QMMA" in sass["ada"], "Ada emits `QMMA` from `mma.sync` in e4m3")

    if execute:
        result = run(cuda / "fatwheels_test_gpu", check=False)
        report(result.returncode == 0, "every generation the driver loads matches serial", result.stdout.splitlines()[0])


def check_rocm(execute: bool) -> None:
    rocm = build("rocm")
    for unit, instruction in (("rocm", "v_fmac_f32"), ("cdna3", "v_mfma_f32_16x16x16_bf16")):
        obj = next((rocm / "CMakeFiles").glob(f"fatwheels_{unit}_.dir/src/capabilities/{unit}.hip.o"))
        bundle, code = rocm / f"{unit}.hipfb", rocm / f"{unit}.co"
        run("/opt/rocm/llvm/bin/llvm-objcopy", f"--dump-section=.hip_fatbin={bundle}", obj)
        run("/opt/rocm/llvm/bin/clang-offload-bundler", "--unbundle", "--type=o", f"--input={bundle}",
            "--targets=hipv4-amdgcn-amd-amdhsa--gfx942", f"--output={code}")
        listing = run("/opt/rocm/llvm/bin/llvm-objdump", "-d", "--mcpu=gfx942", code).stdout
        report(instruction in listing, f"{unit} emits `{instruction}` for gfx942")
    if execute:
        result = run(rocm / "fatwheels_test_gpu", check=False)
        report(result.returncode == 0, "every generation the driver loads matches serial", result.stdout.splitlines()[0])


if __name__ == "__main__":
    mode, execute = (sys.argv[1] if len(sys.argv) > 1 else "cpu"), "--run" in sys.argv
    check_sources()
    if mode == "cpu":
        check_cpu()
    if mode == "gpu" and shutil.which("nvcc"):
        check_cuda(execute)
    if mode == "gpu" and Path("/opt/rocm/llvm/bin/clang").exists():
        check_rocm(execute)
    print(f"{len(failures)} failed" if failures else "all passed")
    sys.exit(1 if failures else 0)
