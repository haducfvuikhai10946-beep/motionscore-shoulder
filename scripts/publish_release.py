"""Publish the reviewed local commit and verified assets; no tokens in arguments/files."""
from __future__ import annotations
import argparse,hashlib,json,shutil,subprocess
from pathlib import Path


def find_release(gh,repo,tag,run):
    """The tag endpoint can return 404 for drafts; list drafts as the owner."""
    response=run(gh,"api","--paginate","--slurp",f"repos/{repo}/releases?per_page=100",capture=True)
    pages=json.loads(response.stdout)
    matches=[release for page in pages for release in page if release["tag_name"]==tag]
    if len(matches)>1:
        raise SystemExit("Multiple releases use "+tag+"; select the intended draft before continuing")
    return matches[0] if matches else None


def asset_is_verified(actual,expected):
    return (actual.get("state")=="uploaded" and actual.get("size")==expected["bytes"]
            and actual.get("digest")=="sha256:"+expected["sha256"])


def main():
    p=argparse.ArgumentParser();p.add_argument("--gh",default="gh")
    p.add_argument("--assets",type=Path)
    p.add_argument("--verify-only",action="store_true",help="Verify publication configuration and asset hashes without login or network writes")
    args=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    publication=json.loads((root/"configs/publication.json").read_text("utf-8"))
    repo,tag=publication["repository"],publication["tag"]
    owner=repo.split("/")[0]
    if len(repo.split("/"))!=2 or not owner or not repo.split("/")[1]:
        raise SystemExit("Invalid publication repository")
    gh=shutil.which(args.gh) or args.gh
    def run(*command,check=True,capture=False):
        result=subprocess.run([str(v) for v in command],cwd=root,text=True,capture_output=capture)
        if check and result.returncode:
            if capture:print(result.stderr.strip())
            raise SystemExit("Command failed; correct the reported issue and rerun. No force push or asset overwrite was attempted.")
        return result
    manifest=json.loads((root/"configs/release_assets.json").read_text("utf-8"))
    assets=(args.assets or root/"release-assets/packages").resolve()
    for item in manifest["assets"]:
        expected_url=f"https://github.com/{repo}/releases/download/{tag}/"+item["name"]
        if item["url"]!=expected_url:raise SystemExit("Asset download URL does not match publication repository: "+item["name"])
        path=(assets/item["name"]).resolve()
        if not path.is_relative_to(assets) or Path(item["name"]).name!=item["name"]:raise SystemExit("Unsafe asset path")
        digest=hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda:stream.read(2**20),b""):digest.update(block)
        if digest.hexdigest()!=item["sha256"]:raise SystemExit("Checksum mismatch: "+item["name"])
        print("Verified",item["name"],flush=True)
    for name in ["CITATION.cff","pyproject.toml"]:
        if "https://github.com/"+repo not in (root/name).read_text("utf-8"):
            raise SystemExit("Repository metadata mismatch: "+name)
    if args.verify_only:
        print("Publication preflight passed for "+repo+" ("+tag+"); no remote operations performed.",flush=True)
        return
    run(gh,"auth","status")
    account=run(gh,"api","user","--jq",".login",capture=True).stdout.strip()
    if account!=owner:raise SystemExit("Login must be "+owner+"; current account: "+account)
    if run("git","status","--porcelain",capture=True).stdout.strip():raise SystemExit("Review and commit local changes before publishing")
    if run("git","branch","--show-current",capture=True).stdout.strip()!="main":raise SystemExit("Expected local main branch")
    run(gh,"auth","setup-git")
    existing=run(gh,"repo","view",repo,"--json","nameWithOwner",capture=True,check=False)
    if existing.returncode:
        run(gh,"repo","create",repo,"--public","--description",
            "Reproducible phase-aware shoulder-raise profiling and MotionScore reference-deviation scoring")
    origin=run("git","remote","get-url","origin",capture=True,check=False)
    if origin.returncode:run("git","remote","add","origin","https://github.com/"+repo+".git")
    elif origin.stdout.strip() not in ["https://github.com/"+repo+".git","git@github.com:"+repo+".git"]:raise SystemExit("Unexpected origin URL")
    run("git","push","--set-upstream","origin","main")
    remote=find_release(gh,repo,tag,run)
    if remote is None:
        run(gh,"release","create",tag,"--repo",repo,"--target","main","--draft","--title",
            "MotionScore v1.0.0: reproducible core and de-identified study data","--notes-file",root/"docs/RELEASE_NOTES.md")
        remote=find_release(gh,repo,tag,run)
        if remote is None:raise SystemExit("Created draft could not be located; stop to avoid creating a duplicate")
    release_id=remote["id"]
    endpoint=f"repos/{repo}/releases/{release_id}"
    print("Using "+("draft" if remote["draft"] else "published")+" release "+str(release_id)+" for "+tag,flush=True)
    for item in manifest["assets"]:
        remote=json.loads(run(gh,"api",endpoint,capture=True).stdout)
        matched=next((a for a in remote["assets"] if a["name"]==item["name"]),None)
        if matched:
            if asset_is_verified(matched,item):
                print("Already uploaded and verified:",item["name"],flush=True);continue
            raise SystemExit("Existing asset needs manual checksum review; it was not overwritten: "+item["name"])
        print("Uploading "+item["name"]+" ("+str(round(item["bytes"]/2**20,1))+" MiB)",flush=True)
        run(gh,"release","upload",tag,assets/item["name"],"--repo",repo)
        remote=json.loads(run(gh,"api",endpoint,capture=True).stdout)
        actual=next((a for a in remote["assets"] if a["name"]==item["name"]),None)
        if actual is None or not asset_is_verified(actual,item):
            raise SystemExit("Uploaded asset could not be verified; release remains a draft: "+item["name"])
        print("Remote SHA-256 verified: "+item["name"],flush=True)
    remote=json.loads(run(gh,"api",endpoint,capture=True).stdout)
    for item in manifest["assets"]:
        actual=next(a for a in remote["assets"] if a["name"]==item["name"])
        if not asset_is_verified(actual,item):raise SystemExit("Remote asset verification failed: "+item["name"])
    run(gh,"api","--method","PATCH",endpoint,"-F","draft=false","--jq",".html_url")
    print("Published https://github.com/"+repo,flush=True)
    print("Release https://github.com/"+repo+"/releases/tag/"+tag,flush=True)

if __name__=="__main__":main()
