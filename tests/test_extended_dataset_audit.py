from hemfm.extended_dataset_audit import SOURCES, _exam_key


def test_ev9v_exam_proxy_removes_video_suffix_only():
    assert _exam_key("20210101_120000_001_640x480_000") == "20210101_120000"
    assert _exam_key("20210101_120000_640x480_000") == "20210101_120000"


def test_ev9v_exam_proxy_preserves_non_suffix_digits():
    assert _exam_key("patient_012_view_A4C") == "patient_012_view_A4C"


def test_controlled_access_sources_use_authoritative_project_pages():
    assert SOURCES["echonet_pediatric"] == "https://echonet.github.io/pediatric/"
    assert SOURCES["echoview"] == "https://physionet.org/content/echoview/0.1/"

