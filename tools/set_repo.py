"""Point the notebooks and README at the real GitHub repo. Run once, from the repo root:

    python tools/set_repo.py your-github-name/your-repo-name
"""
import glob
import sys

if len(sys.argv) != 2 or sys.argv[1].count("/") != 1:
    sys.exit("usage: python tools/set_repo.py OWNER/REPO")
for path in glob.glob("notebooks/*.ipynb") + ["README.md"]:
    text = open(path).read()
    new = text.replace("OWNER/ml4math-mini-course", sys.argv[1])
    if new != text:
        open(path, "w").write(new)
        print("updated", path)
