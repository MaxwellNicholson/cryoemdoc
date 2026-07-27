from __future__ import annotations

from cryoemdoc.model_download import release_asset_url


def test_release_asset_url_defaults_to_project_release_asset():
    assert (
        release_asset_url()
        == "https://github.com/MaxwellNicholson/cryoemdoc/releases/download/v0.1.0/cryoemdoc-models-v0.1.0.zip"
    )


def test_release_asset_url_accepts_custom_archive_name():
    assert (
        release_asset_url(version="v1.2.3", repository="owner/repo", archive_name="models.zip")
        == "https://github.com/owner/repo/releases/download/v1.2.3/models.zip"
    )
