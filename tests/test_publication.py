"""Regression coverage for draft lookup without live GitHub writes."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

spec=importlib.util.spec_from_file_location("publish_release",Path(__file__).resolve().parents[1]/"scripts/publish_release.py")
publication=importlib.util.module_from_spec(spec)
spec.loader.exec_module(publication)


def test_find_draft_on_later_page_without_tag_endpoint():
    draft={"id":42,"tag_name":"v1.0.0","draft":True,"assets":[]}
    calls=[]
    def run(*args,**kwargs):
        calls.append(args)
        return SimpleNamespace(stdout=json.dumps([[{"id":1,"tag_name":"other"}],[draft]]))
    assert publication.find_release("gh","owner/repo","v1.0.0",run)==draft
    assert len(calls)==1
    assert calls[0][-1]=="repos/owner/repo/releases?per_page=100"
    assert "--paginate" in calls[0] and "--slurp" in calls[0]


def test_empty_release_list_is_the_only_absence_signal():
    def empty(*args,**kwargs):return SimpleNamespace(stdout="[[]]")
    assert publication.find_release("gh","owner/repo","v1.0.0",empty) is None
    def denied(*args,**kwargs):raise SystemExit("GitHub access denied")
    with pytest.raises(SystemExit,match="access denied"):
        publication.find_release("gh","owner/repo","v1.0.0",denied)


def test_duplicate_tag_drafts_stop_before_mutation():
    def run(*args,**kwargs):
        return SimpleNamespace(stdout=json.dumps([[{"id":1,"tag_name":"v1.0.0"},{"id":2,"tag_name":"v1.0.0"}]]))
    with pytest.raises(SystemExit,match="Multiple releases"):
        publication.find_release("gh","owner/repo","v1.0.0",run)


def test_resume_requires_uploaded_state_size_and_digest():
    expected={"bytes":100,"sha256":"abc"}
    actual={"state":"uploaded","size":100,"digest":"sha256:abc"}
    assert publication.asset_is_verified(actual,expected)
    for change in [{"state":"starter"},{"size":0},{"digest":None},{"digest":"sha256:wrong"}]:
        assert not publication.asset_is_verified(dict(actual,**change),expected)
