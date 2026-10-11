"""
Smoke test for the demo app (app/streamlit_app.py).

WHAT: runs the whole app script headlessly and checks nothing raises.
WHY:  the app is what we open in the viva. A typo in it would only show up as a red
      error on screen in front of the examiner, so it is worth a test.
NOTE: this trains the ensemble, so it is the slowest test in the suite (~40s).
      It is skipped automatically if the dataset has not been built yet.
"""

import pytest

from src.config import COMBINED_CSV, PROJECT_ROOT

# An absolute path, because the test may be run from any directory.
APP = str(PROJECT_ROOT / "app" / "streamlit_app.py")


@pytest.mark.skipif(not COMBINED_CSV.exists(), reason="dataset not built")
def test_the_app_runs_and_shows_all_seven_sections():
    """Every section must render, and none of them may raise."""
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(APP, default_timeout=600).run()
    assert not app.exception, [str(error.value) for error in app.exception]
    headers = [header.value for header in app.subheader]
    assert len(headers) == 7
    for number in range(1, 8):
        assert any(header.startswith(f"{number}.") for header in headers)


@pytest.mark.skipif(not COMBINED_CSV.exists(), reason="dataset not built")
def test_the_example_scam_message_is_flagged():
    """
    The message the app opens with must be caught.

    If the default example were missed, the demo would start by failing.
    """
    from streamlit.testing.v1 import AppTest
    from src.config import FRAUD_THRESHOLD
    app = AppTest.from_file(APP, default_timeout=600).run()
    ensemble = [metric for metric in app.metric if metric.label == "Ensemble (F4)"]
    assert ensemble, "the ensemble score is not displayed"
    assert float(ensemble[0].value) >= FRAUD_THRESHOLD


@pytest.mark.skipif(not COMBINED_CSV.exists(), reason="dataset not built")
def test_the_attack_button_works():
    """Section 6 runs the F8 attacks live, so clicking it must not break."""
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(APP, default_timeout=600).run()
    app.button[0].click().run()
    assert not app.exception, [str(error.value) for error in app.exception]
