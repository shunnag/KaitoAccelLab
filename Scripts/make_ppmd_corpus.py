#!/usr/bin/env python3
"""Build deterministic PPMd H/I decoding workloads with black-box 7zz."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("destination", type=Path)
parser.add_argument("--image-size", type=int, default=2048)
parser.add_argument("--kaito-kit", type=Path,
                    default=Path(__file__).resolve().parents[2] / "KaitoKit")
args = parser.parse_args()
root = args.destination.resolve()
if root.exists() and (not root.is_dir() or any(root.iterdir())):
    parser.error("destination must be new or empty")
root.mkdir(parents=True, exist_ok=True)
repository = args.kaito_kit.resolve()
spec = importlib.util.spec_from_file_location("fixtures", repository / "Scripts/fixtures/make-rar4.py")
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)
rng = random.Random(2026102903)
words = (b"archive reader entry stream buffer checksum decode file directory metadata "
         b"window block table symbol length distance offset byte count format header "
         b"compression measurement sample round baseline branch identity timestamp").split()
text = bytearray()
while len(text) < 16 * 1024 * 1024:
    text.extend(rng.choice(words) + b" ")
(root / "text.txt").write_bytes(text[:16 * 1024 * 1024])
inputs = ["text.txt"] + [f"page-{page}.bmp" for page in range(3)]
for page in range(3):
    name = f"page-{page}.bmp"
    (root / name).write_bytes(fixtures.image_bmp(args.image_size, args.image_size, page))
    subprocess.run(["sips", "-s", "format", "jpeg", str(root / name),
                    "--out", str(root / f"page-{page}.jpg")], check=True,
                   stdout=subprocess.DEVNULL)
    inputs.append(f"page-{page}.jpg")
manifest = {"image_size": args.image_size, "text_seed": 2026102903, "inputs": {}, "archives": {}}
for name in inputs:
    data = (root / name).read_bytes()
    manifest["inputs"][name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
for name, members in [("text", ["text.txt"]), ("color", [f"page-{page}.bmp" for page in range(3)]),
                      ("jpeg", [f"page-{page}.jpg" for page in range(3)])]:
    for extension, method in [("7z", "-m0=PPMd"), ("zip", "-mm=PPMd")]:
        archive = root / f"{name}.{extension}"
        subprocess.run(["7zz", "a", "-t" + extension, method, "-mmt=1", "-bso0", "-bsp0",
                        str(archive), *members], cwd=root, check=True)
        data = archive.read_bytes()
        manifest["archives"][archive.name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                                             "entries": members}
for path in root.iterdir():
    if path.is_file() and path.suffix != ".7z" and path.suffix != ".zip":
        os.utime(path, (1700000000, 1700000000))
(root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
print(json.dumps(manifest, indent=2, sort_keys=True))
