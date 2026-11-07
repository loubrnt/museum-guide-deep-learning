"""Download a small set of NON-painting photos to test what the guide does when it is not shown a painting.

    python scripts/fetch_non_paintings.py            # about 400 MB to download, a minute or two

Sources (research datasets of ordinary photos, used here only to test locally; nothing is committed):

- `caltech_objects`: Caltech-101 objects and animals (Hugging Face `flwrlabs/caltech101`, about 120 MB);
- `caltech_faces`: the two "faces" classes of Caltech-101 (people's faces);
- `imagenette`: ten ImageNet classes (objects, a dog, a church, a parachute...; fast.ai `imagenette2-160`, about 100 MB);
- `dtd`: the test split of the Describable Textures Dataset (surfaces and materials, Hugging Face `tanganke/dtd`, about 180 MB).

For each source, two disjoint draws with a fixed seed: 300 images for `test` and 150 for `val` (the `val` draw is used
to choose a threshold, the `test` draw to report). Each image gets the SAME treatment as the paintings of the data
subset (`project/make_wikiart_subset.py`): bicubic resize and centre crop to 128 x 128, JPEG quality 90, so that the
only difference between a painting and a photo is its content, not its resolution or compression.

Writes `data/ood/<source>/<val|test>/*.jpg` and `data/ood/manifest.json` (`data/` is not versioned).
"""
import argparse
import io
import json
import random
import tarfile
import urllib.request
from pathlib import Path

from PIL import Image, ImageOps

OUT = Path("data/ood")
SIZE, QUALITY, SEED = 128, 90, 123
N_TEST, N_VAL = 300, 150
IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"


def save(img: Image.Image, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    ImageOps.fit(img.convert("RGB"), (SIZE, SIZE), method=Image.Resampling.BICUBIC).save(dst, "JPEG", quality=QUALITY)


def draw(pool: list, seed: int = SEED) -> dict[str, list]:
    """Two disjoint random draws from a pool: `test` first, then `val`."""
    rng = random.Random(seed)
    items = list(pool)
    rng.shuffle(items)
    return {"test": items[:N_TEST], "val": items[N_TEST : N_TEST + N_VAL]}


def hf_source(name: str, repo: str, split: str, keep) -> dict[str, list]:
    from datasets import load_dataset

    ds = load_dataset(repo, split=split)
    names = ds.features["label"].names
    pool = [i for i, lab in enumerate(ds["label"]) if keep(names[lab])]
    chosen = draw(pool)
    out = {}
    for subset, idxs in chosen.items():
        files = []
        for i in idxs:
            dst = OUT / name / subset / f"{name}_{i:05d}.jpg"
            save(ds[i]["image"], dst)
            files.append(str(dst))
        out[subset] = files
    return out


def imagenette_source() -> dict[str, list]:
    raw = OUT / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    archive = raw / "imagenette2-160.tgz"
    if not archive.exists():
        print("downloading imagenette (about 100 MB)...", flush=True)
        urllib.request.urlretrieve(IMAGENETTE_URL, archive)
    with tarfile.open(archive) as tar:
        members = sorted((m for m in tar.getmembers() if m.isfile() and m.name.lower().endswith(".jpeg")), key=lambda m: m.name)
        chosen = draw(members)
        out = {}
        for subset, ms in chosen.items():
            files = []
            for m in ms:
                dst = OUT / "imagenette" / subset / (Path(m.name).stem + ".jpg")
                save(Image.open(io.BytesIO(tar.extractfile(m).read())), dst)
                files.append(str(dst))
            out[subset] = files
    return out


def main():
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    faces = ("faces", "faces_easy")
    manifest = {"size": SIZE, "jpeg_quality": QUALITY, "seed": SEED, "n_test": N_TEST, "n_val": N_VAL, "sources": {}}
    print("caltech101...", flush=True)
    manifest["sources"]["caltech_objects"] = hf_source("caltech_objects", "flwrlabs/caltech101", "train", lambda n: n not in faces)
    manifest["sources"]["caltech_faces"] = hf_source("caltech_faces", "flwrlabs/caltech101", "train", lambda n: n in faces)
    print("dtd...", flush=True)
    manifest["sources"]["dtd"] = hf_source("dtd", "tanganke/dtd", "test", lambda n: True)
    manifest["sources"]["imagenette"] = imagenette_source()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    for name, subsets in manifest["sources"].items():
        print(f"{name:16s} test {len(subsets['test'])}  val {len(subsets['val'])}")


if __name__ == "__main__":
    main()
