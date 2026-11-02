"""Publish the reviewed local commit and verified assets; no tokens in arguments/files."""
from __future__ import annotations
import argparse,hashlib,json,shutil,subprocess
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument("--gh",default="gh")
    p.add_argument("--assets",type=Path);args=p.parse_args()
    root=Path(__file__).resolve().parents[1];repo="jackcr7900/motionscore-shoulder";tag="v1.0.0"
    gh=shutil.which(args.gh) or args.gh
    def run(*command,check=True,capture=False):
        result=subprocess.run([str(v) for v in command],cwd=root,text=True,capture_output=capture)
        if check and result.returncode:
            if capture:print(result.stderr.strip())
            raise SystemExit("Command failed; correct the reported issue and rerun. No force push or asset overwrite was attempted.")
        return result
    run(gh,"auth","status")
    account=run(gh,"api","user","--jq",".login",capture=True).stdout.strip()
    if account!="jackcr7900":raise SystemExit("Login must be jackcr7900; current account: "+account)
    if run("git","status","--porcelain",capture=True).stdout.strip():raise SystemExit("Review and commit local changes before publishing")
    if run("git","branch","--show-current",capture=True).stdout.strip()!="main":raise SystemExit("Expected local main branch")
    manifest=json.loads((root/"configs/release_assets.json").read_text("utf-8"))
    assets=(args.assets or root/"release-assets/packages").resolve()
    for item in manifest["assets"]:
        path=(assets/item["name"]).resolve()
        if not path.is_relative_to(assets) or Path(item["name"]).name!=item["name"]:raise SystemExit("Unsafe asset path")
        digest=hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda:stream.read(2**20),b""):digest.update(block)
        if digest.hexdigest()!=item["sha256"]:raise SystemExit("Checksum mismatch: "+item["name"])
        print("Verified",item["name"],flush=True)
    run(gh,"auth","setup-git")
    existing=run(gh,"repo","view",repo,"--json","nameWithOwner",capture=True,check=False)
    if existing.returncode:
        run(gh,"repo","create",repo,"--public","--description",
            "Reproducible phase-aware shoulder-raise profiling and MotionScore reference-deviation scoring")
    origin=run("git","remote","get-url","origin",capture=True,check=False)
    if origin.returncode:run("git","remote","add","origin","https://github.com/"+repo+".git")
    elif origin.stdout.strip() not in ["https://github.com/"+repo+".git","git@github.com:"+repo+".git"]:raise SystemExit("Unexpected origin URL")
    run("git","push","--set-upstream","origin","main")
    release=run(gh,"api",f"repos/{repo}/releases/tags/{tag}",capture=True,check=False)
    if release.returncode:
        run(gh,"release","create",tag,"--repo",repo,"--target","main","--draft","--title",
            "MotionScore v1.0.0: reproducible core and de-identified study data","--notes-file",root/"docs/RELEASE_NOTES.md")
        release=run(gh,"api",f"repos/{repo}/releases/tags/{tag}",capture=True)
    remote=json.loads(release.stdout)
    for item in manifest["assets"]:
        matched=next((a for a in remote["assets"] if a["name"]==item["name"]),None)
        if matched:
            if matched.get("digest")=="sha256:"+item["sha256"]:
                print("Already uploaded and verified:",item["name"],flush=True);continue
            raise SystemExit("Existing asset needs manual checksum review; it was not overwritten: "+item["name"])
        run(gh,"release","upload",tag,assets/item["name"],"--repo",repo)
    remote=json.loads(run(gh,"api",f"repos/{repo}/releases/tags/{tag}",capture=True).stdout)
    for item in manifest["assets"]:
        actual=next(a for a in remote["assets"] if a["name"]==item["name"])
        if actual["size"]!=item["bytes"]:raise SystemExit("Remote size mismatch")
        if actual.get("digest") and actual["digest"]!="sha256:"+item["sha256"]:raise SystemExit("Remote digest mismatch")
    run(gh,"release","edit",tag,"--repo",repo,"--draft=false")
    print("Published https://github.com/"+repo,flush=True)
    print("Release https://github.com/"+repo+"/releases/tag/"+tag,flush=True)

if __name__=="__main__":main()
