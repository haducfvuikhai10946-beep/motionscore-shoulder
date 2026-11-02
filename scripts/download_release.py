"""Download named release assets and verify SHA-256 before safe extraction."""
import argparse,hashlib,json,shutil,time,urllib.request,zipfile
from pathlib import Path


def download(url,path,expected):
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest()==expected:return
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url,timeout=60) as response,path.open("wb") as dest:shutil.copyfileobj(response,dest)
            if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise RuntimeError("SHA-256 mismatch")
            return
        except Exception:
            if attempt==3:raise
            time.sleep(2**attempt)


def main():
    p=argparse.ArgumentParser();p.add_argument("--kind",choices=["models","dataset","videos","all"],default="models")
    p.add_argument("--output",type=Path,default=Path("downloads"));args=p.parse_args()
    manifest=json.loads((Path(__file__).resolve().parents[1]/"configs/release_assets.json").read_text())
    args.output.mkdir(parents=True,exist_ok=True)
    for item in manifest["assets"]:
        if args.kind!="all" and item["kind"]!=args.kind:continue
        target=args.output/item["name"];download(item["url"],target,item["sha256"])
        with zipfile.ZipFile(target) as archive:
            for member in archive.infolist():
                path=(args.output/member.filename).resolve()
                if not path.is_relative_to(args.output.resolve()):raise RuntimeError("Unsafe archive path")
                if (member.external_attr>>16)&0o170000==0o120000:raise RuntimeError("Symlinks not allowed")
            archive.extractall(args.output)
        print("Verified and extracted",target.name,flush=True)

if __name__=="__main__":main()
