
from pathlib import Path
from zipfile import ZipFile, ZIP_STORED

checkpoint_dir = Path("checkpoints")

for name in ["Best", "Final", "model"]:
    source = checkpoint_dir / f"{name}.pkl" / name
    output = checkpoint_dir / f"{name}.pth"

    if not source.is_dir():
        print(f"Missing extracted folder: {source}")
        continue

    with ZipFile(output, "w", compression=ZIP_STORED) as archive:
        for file in source.rglob("*"):
            if file.is_file():
                # Preserve the original PyTorch archive root folder.
                archive.write(file, file.relative_to(source.parent))

    # Verify archive integrity and required files.
    with ZipFile(output, "r") as archive:
        bad_file = archive.testzip()
        names = archive.namelist()

        if bad_file:
            raise RuntimeError(f"{name}: corrupted entry: {bad_file}")

        required = [
            f"{name}/data.pkl",
            f"{name}/version",
            f"{name}/byteorder",
        ]

        missing = [item for item in required if item not in names]
        if missing:
            raise RuntimeError(f"{name}: missing entries: {missing}")

    print(f"Reconstructed and verified: {output}")
