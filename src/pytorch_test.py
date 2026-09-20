import os
import sys
import glob
import torch


print("=" * 70)
print("PYTORCH PACKAGE AUDIT")
print("=" * 70)

print()
print("Python:")
print(sys.executable)

print()
print("Python version:")
print(sys.version)


# ============================================================
# LOCATE TORCH
# ============================================================

torch_root = os.path.dirname(torch.__file__)

torch_lib = os.path.join(
    torch_root,
    "lib"
)


print()
print("Torch directory:")
print(torch_root)

print()
print("Torch directory exists:")
print(os.path.isdir(torch_root))

print()
print("Torch lib directory:")
print(torch_lib)

print()
print("Torch lib directory exists:")
print(os.path.isdir(torch_lib))


# ============================================================
# LIST TORCH DLL FILES
# ============================================================

print()
print("DLL files in torch\\lib:")

dll_files = sorted(
    glob.glob(
        os.path.join(
            torch_lib,
            "*.dll"
        )
    )
)

if not dll_files:

    print("  No DLL files found.")

else:

    for dll in dll_files:

        size = os.path.getsize(dll)

        print(
            f"  {os.path.basename(dll)}"
            f"  ({size:,} bytes)"
        )


# ============================================================
# CHECK TORCH PACKAGE METADATA
# ============================================================

site_packages = os.path.dirname(
    torch_root
)

metadata_patterns = [
    os.path.join(
        site_packages,
        "torch-*.dist-info"
    ),
    os.path.join(
        site_packages,
        "torch-*.egg-info"
    ),
]

print()
print("Torch package metadata:")

metadata_found = []

for pattern in metadata_patterns:

    metadata_found.extend(
        glob.glob(pattern)
    )

if metadata_found:

    for item in metadata_found:
        print(" ", item)

else:

    print("  No torch metadata directory found.")


# ============================================================
# CHECK TORCH.PY
# ============================================================

torch_init = os.path.join(
    torch_root,
    "__init__.py"
)

print()
print("torch __init__.py exists:")
print(os.path.isfile(torch_init))


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 70)
print("PYTORCH PACKAGE AUDIT COMPLETE")
print("=" * 70)

