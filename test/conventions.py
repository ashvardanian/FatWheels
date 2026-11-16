"""The rules of FatWheels' two capability flags, each as a test over the builds, objects and images that prove it.

    pytest test/conventions.py -m cpu            # in `fatwheels:cpu`: x86 natively and under SDE, AArch64 under QEMU
    pytest test/conventions.py -m "cuda or rocm" # in `fatwheels:gpu`: compiles, kernel entries, instructions
    pytest test/conventions.py --run-devices     # also run kernels on the GPUs this machine has

Builds are session fixtures, configured from the presets once and shared by every test that reads them.
"""

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

root = Path(__file__).resolve().parent.parent
include_flags = ["-I", root / "c", "-I", root / "include"]
rocm_bin = Path("/opt/rocm/llvm/bin")
sde = shutil.which("sde64") or os.environ.get("SDE")

needs_nvcc = pytest.mark.skipif(not shutil.which("nvcc"), reason="no NVCC")
needs_rocm = pytest.mark.skipif(not (rocm_bin / "clang").exists(), reason="no ROCm Clang")
needs_qemu = pytest.mark.skipif(not shutil.which("qemu-aarch64"), reason="no AArch64 QEMU")
needs_sde = pytest.mark.skipif(not sde, reason="no Intel SDE")


def run(*command, check: bool = True, stdin: str | None = None) -> subprocess.CompletedProcess:
    result = subprocess.run([str(part) for part in command], capture_output=True, text=True, input=stdin, check=False)
    if check:
        assert result.returncode == 0, f"{' '.join(map(str, command))}\n{result.stdout}{result.stderr}"
    return result


@pytest.fixture(scope="session")
def build_example():
    """Builds `example/` as its own project, as a downstream user would, once per vendor."""
    trees: dict[str, Path] = {}

    def tree_of(name: str, *definitions: str) -> Path:
        if name not in trees:
            tree = root / f"build_example_{name}"
            arguments = [
                argument
                for definition in ("CMAKE_C_COMPILER=clang-23", *definitions)
                for argument in ("-D", definition)
            ]
            run("cmake", "-S", root / "example", "-B", tree, "-G", "Ninja", *arguments)
            run("cmake", "--build", tree)
            trees[name] = tree
        return trees[name]

    return tree_of


def definitions_of(tree: Path, source: str) -> dict[str, str]:
    """The `-D` definitions CMake passes to the unit compiling `source`."""
    for entry in json.loads((tree / "compile_commands.json").read_text()):
        if entry["file"].endswith(source):
            return dict(re.findall(r"-D(\w+)=(\S+)", entry["command"]))
    raise LookupError(source)


def preprocess(compiler: str, *flags: str, source: str = "#include <fatwheels/types.h>\n") -> dict[str, str]:
    """The macros `source` sees under `flags`, with no build system around it."""
    output = run(compiler, *flags, *include_flags, "-dM", "-E", "-x", "c", "-", stdin=source).stdout
    return dict(re.findall(r"#define (FATWHEELS_\w+) (\S+)", output))


def defined_kernels(object_path: Path) -> set[str]:
    return set(re.findall(r" [Tt] (fw_matmul_f32_\w+)", run("nm", object_path).stdout))


def unit_object(tree: Path, unit: str, extension: str = "c", target: str = "fatwheels_static") -> Path:
    return tree / f"CMakeFiles/{target}.dir/src/capabilities/{unit}.{extension}.o"


def cubin_symbols(path: Path) -> tuple[set[str], set[str]]:
    """The kernel entries and `__device__` globals of every cubin in an object or fatbin."""
    entries, globals_ = set(), set()
    in_symbols = False
    for line in run("cuobjdump", "-elf", path).stdout.splitlines():
        if line.startswith(".section"):
            in_symbols = line.startswith(".section .symtab")
        elif in_symbols and len(fields := line.split()) > 4:
            if fields[3] in ("0x2", "0x12"):
                entries.add(fields[-1])
            elif fields[3] in ("0x1", "0x11"):
                globals_.add(fields[-1])
    return entries, globals_


def amd_code_object(object_path: Path, scratch: Path, target: str = "gfx942") -> Path:
    """The `target` code object of a HIP object's offload bundle."""
    bundle, code = scratch / f"{object_path.stem}.hipfb", scratch / f"{object_path.stem}.co"
    run(rocm_bin / "llvm-objcopy", f"--dump-section=.hip_fatbin={bundle}", object_path)
    run(
        rocm_bin / "clang-offload-bundler",
        "--unbundle",
        "--type=o",
        f"--input={bundle}",
        f"--targets=hipv4-amdgcn-amd-amdhsa--{target}",
        f"--output={code}",
    )
    return code


# Sources


def test_arch_flags_read_no_target():
    types = (root / "c/fatwheels/types.h").read_text().splitlines()
    readers = [line for line in types if line.startswith("#define FATWHEELS_ARCH_") and "TARGET" in line]
    assert not readers


def test_no_region_gated_by_several_targets():
    gated = [
        f"{path.name}: {line}"
        for path in (root / "include/fatwheels").iterdir()
        for line in path.read_text().splitlines()
        if re.match(r"#if FATWHEELS_TARGET_\w+ \|\|", line)
    ]
    assert not gated


def test_example_names_no_library_internal():
    sources = "".join(path.read_text() for path in (root / "example").iterdir() if path.is_file())
    assert not set(re.findall(r"FATWHEELS_ARCH_\w+|\bfw_\w+_\b|fatwheels_\w+_\b", sources))


# CPU units and flags

cpu_units = ["serial", "haswell", "skylake", "sapphireamx"]
cpu_instructions = [
    ("haswell", r"vfmadd\w+ps\s[^\n]*%ymm"),
    ("skylake", r"vfmadd\w+ps\s[^\n]*%zmm"),
    ("sapphireamx", r"tdpbf16ps"),
]


@pytest.mark.cpu
@pytest.mark.parametrize("unit", cpu_units)
def test_unit_defines_its_own_kernel_only(build, unit):
    assert defined_kernels(unit_object(build("clang-floor"), unit)) == {f"fw_matmul_f32_{unit}"}


@pytest.mark.cpu
@pytest.mark.parametrize("unit, instruction", cpu_instructions, ids=[unit for unit, _ in cpu_instructions])
def test_unit_emits_its_instruction(build, unit, instruction):
    assert re.search(instruction, run("objdump", "-d", unit_object(build("clang-floor"), unit)).stdout)


@pytest.mark.cpu
def test_probe_raises_what_source_cannot_see(build):
    assert preprocess("clang-23")["FATWHEELS_ARCH_X8664_HASWELL_"] == "0"
    assert definitions_of(build("clang-floor"), "src/capabilities/haswell.c")["FATWHEELS_ARCH_X8664_HASWELL_"] == "1"


@pytest.mark.cpu
def test_source_detects_what_the_floor_enables():
    assert preprocess("gcc-15", "-march=x86-64-v3")["FATWHEELS_ARCH_X8664_HASWELL_"] == "1"


@pytest.mark.cpu
def test_raising_a_capability_opens_base_blocks_not_base_kernels():
    macros = preprocess("clang-23", "-DFATWHEELS_ARCH_X8664_SKYLAKE_=1")
    assert (macros["FATWHEELS_TARGET_SKYLAKE"], macros["FATWHEELS_TARGET_HASWELL"]) == ("1", "0")
    assert macros["FATWHEELS_ARCH_X8664_HASWELL_"] == "1"


@pytest.mark.cpu
def test_dropped_target_keeps_its_arch_flag(build):
    tree = build("clang-without-amx")
    flags = definitions_of(tree, "src/fatwheels.c")
    assert (flags["FATWHEELS_ARCH_X8664_SAPPHIREAMX_"], flags["FATWHEELS_TARGET_SAPPHIREAMX"]) == ("1", "0")
    assert not unit_object(tree, "sapphireamx").exists()


@pytest.mark.cpu
def test_monoversioned_build_keeps_what_the_floor_implies(build):
    flags = definitions_of(build("gcc-monoversioned"), "src/fatwheels.c")
    targets = {
        capability: flags[f"FATWHEELS_TARGET_{capability}"] for capability in ("HASWELL", "SKYLAKE", "SAPPHIREAMX")
    }
    assert targets == {"HASWELL": "1", "SKYLAKE": "0", "SAPPHIREAMX": "0"}
    assert flags["FATWHEELS_ARCH_X8664_SKYLAKE_"] == "1"


@pytest.mark.cpu
def test_monoversioned_tables_hold_one_kernel(build):
    assert "dispatched 0x2" in run(build("gcc-monoversioned") / "fatwheels_test").stdout


@pytest.mark.cpu
def test_header_only_user_needs_no_build_system(tmp_path):
    run("clang-23", "-O2", *include_flags, root / "test/header_only.c", "-o", tmp_path / "header_only")
    run(tmp_path / "header_only")


@pytest.mark.cpu
def test_hand_set_target_without_flags_fails_to_link(tmp_path):
    result = run(
        "clang-23",
        "-O2",
        "-DFATWHEELS_TARGET_SAPPHIREAMX=1",
        *include_flags,
        root / "test/header_only.c",
        "-o",
        tmp_path / "header_only",
        check=False,
    )
    assert result.returncode != 0 and "fw_matmul_f32_sapphireamx" in result.stderr


@pytest.mark.cpu
def test_declarations_only_unit_links_the_library(build):
    run(build("clang-floor") / "fatwheels_test_declarations")


x86_emulators = [
    pytest.param([], id="native"),
    pytest.param([sde, "-skx", "--"], id="sde-skylake", marks=needs_sde),
    pytest.param([sde, "-spr", "--"], id="sde-sapphire-rapids", marks=needs_sde),
]
arm_cpus = ["max,sve=off,sme=off", "max,sme=off,sve-max-vq=1", "max,sve-max-vq=4"]


@pytest.mark.cpu
@pytest.mark.parametrize("emulator", x86_emulators)
def test_x86_kernels_match_serial(build, emulator):
    run(*emulator, build("clang-floor") / "fatwheels_test")


@pytest.mark.cpu
@needs_qemu
@pytest.mark.parametrize("cpu", arm_cpus)
def test_arm_kernels_match_serial(build, cpu):
    run("qemu-aarch64", "-cpu", cpu, build("aarch64-qemu") / "fatwheels_test")


# CUDA

cuda_generations = ["cuda", "ampere", "ada"]


@pytest.fixture(scope="session")
def cuda_units(build):
    tree = build("cuda")
    return {unit: unit_object(tree, unit, "cu", f"fatwheels_{unit}_") for unit in cuda_generations}


@pytest.mark.cuda
@needs_nvcc
@pytest.mark.parametrize("unit", cuda_generations)
def test_generation_unit_holds_its_own_kernels(cuda_units, unit):
    entries, _ = cubin_symbols(cuda_units[unit])
    owners = {re.search(r"fw_\w+?_(cuda|ampere|ada)_kernel_", entry).group(1) for entry in entries}
    assert owners == {unit}


@pytest.mark.cuda
@needs_nvcc
def test_device_table_stays_with_its_generation(cuda_units):
    assert "fw_halves_ampere_" in cubin_symbols(cuda_units["ampere"])[1]
    assert not cubin_symbols(cuda_units["ada"])[1]


@pytest.mark.cuda
@needs_nvcc
@pytest.mark.parametrize("unit, instruction", [("ampere", "HMMA"), ("ada", "QMMA")])
def test_generation_emits_its_matrix_instruction(cuda_units, unit, instruction):
    assert instruction in run("cuobjdump", "-sass", cuda_units[unit]).stdout


@pytest.mark.cuda
@needs_nvcc
def test_cuda_generations_match_serial(build, request):
    if not request.config.getoption("--run-devices"):
        pytest.skip("needs --run-devices")
    run(build("cuda") / "fatwheels_test_gpu")


@pytest.fixture(scope="session")
def example_cuda(build_example):
    return build_example("cuda", "FATWHEELS_CUDA_ARCH=80;89;120") / "embedded/example_ada.fatbin"


@pytest.mark.cuda
@needs_nvcc
def test_example_fatbin_holds_its_one_kernel(example_cuda):
    assert cubin_symbols(example_cuda) == ({"example_score_ada_kernel"}, set())
    assert "QMMA" in run("cuobjdump", "-sass", example_cuda).stdout


@pytest.mark.cuda
@needs_nvcc
def test_library_target_on_ships_library_kernels(tmp_path):
    fatbin = tmp_path / "example_ada_with_target.fatbin"
    run(
        "nvcc",
        "-fatbin",
        "-arch=sm_89",
        "-DFATWHEELS_TARGET_ADA=1",
        *include_flags,
        root / "example/ada.cu",
        "-o",
        fatbin,
    )
    assert any("fw_matmul_f32_ada" in entry for entry in cubin_symbols(fatbin)[0])


@pytest.mark.cuda
@needs_nvcc
def test_example_runs_through_the_device_services(build_example, example_cuda, request):
    if not request.config.getoption("--run-devices"):
        pytest.skip("needs --run-devices")
    assert "matches serial" in run(example_cuda.parent.parent / "example_gpu").stdout


# ROCm

rocm_instructions = [("rocm", "v_fmac_f32"), ("cdna3", "v_mfma_f32_16x16x16_bf16")]


@pytest.mark.rocm
@needs_rocm
@pytest.mark.parametrize("unit, instruction", rocm_instructions, ids=[unit for unit, _ in rocm_instructions])
def test_rocm_generation_emits_its_instruction(build, tmp_path, unit, instruction):
    code = amd_code_object(unit_object(build("rocm"), unit, "hip", f"fatwheels_{unit}_"), tmp_path)
    assert instruction in run(rocm_bin / "llvm-objdump", "-d", "--mcpu=gfx942", code).stdout


@pytest.mark.rocm
@needs_rocm
def test_example_code_object_holds_its_one_kernel(build_example):
    tree = build_example(
        "rocm",
        "CMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++",
        "CMAKE_PREFIX_PATH=/opt/rocm",
        "FATWHEELS_ROCM_ARCH=gfx942",
    )
    code = tree / "embedded/example_cdna3.hsaco"
    symbols = {
        fields[7]
        for fields in map(str.split, run(rocm_bin / "llvm-readelf", "--symbols", code).stdout.splitlines())
        if len(fields) == 8 and (fields[3] == "FUNC" or fields[7].endswith(".kd"))
    }
    assert symbols == {"example_score_cdna3_kernel", "example_score_cdna3_kernel.kd"}
    assert "v_mfma_f32_16x16x16_bf16" in run(rocm_bin / "llvm-objdump", "-d", "--mcpu=gfx942", code).stdout


@pytest.mark.rocm
@needs_rocm
def test_rocm_generations_match_serial(build, request):
    if not request.config.getoption("--run-devices"):
        pytest.skip("needs --run-devices")
    run(build("rocm") / "fatwheels_test_gpu")


# Example on the CPU


@pytest.mark.cpu
@pytest.mark.parametrize("unit", ["serial", "haswell", "sapphireamx"])
def test_example_unit_carries_its_own_kernel_only(build_example, unit):
    assert defined_kernels(build_example("cpu") / f"CMakeFiles/example_cpu.dir/{unit}.c.o") == {f"fw_matmul_f32_{unit}"}


@pytest.mark.cpu
@pytest.mark.parametrize("emulator", [x86_emulators[0], x86_emulators[2]])
def test_example_matches_serial(build_example, emulator):
    run(*emulator, build_example("cpu") / "example_cpu")
