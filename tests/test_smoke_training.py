import sys
import subprocess
import pytest

@pytest.mark.slow
def test_smoke_training():
    result = subprocess.run([
       sys.executable,
       "-m",
       "ncds.cli.train",
        "--config=configs/ci_train.py",
       ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert result.returncode == 0, result.stderr.decode()

    