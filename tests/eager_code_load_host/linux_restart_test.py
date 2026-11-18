import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

with tempfile.TemporaryDirectory(dir=sys.argv[2]) as tmp:
    image = pathlib.Path(tmp) / "engine716"
    shutil.copy2(sys.argv[1], image)
    env = os.environ.copy()
    env.pop("HIP_ENABLE_DEFERRED_LOADING", None)
    env.pop("STRIX_EAGER_CODE_LOAD", None)
    subprocess.run([str(image), sys.argv[3], ""], env=env, check=True, timeout=20)
