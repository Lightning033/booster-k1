"""
Make booster_robotics_sdk_python importable from THIS interpreter.

Why this is needed: k1_follow_env was created with --system-site-packages, which
deliberately ignores ~/.local so the stray numpy 2.2.6 and torch 2.14 installed
there cannot break ROS2. The Booster SDK is very likely installed in that same
~/.local, so the isolation that protects numpy also hides the SDK. That is why
`python sdk_client/sdk_client.py walk` worked before the venv existed and fails
inside it.

The fix is a symlink for the SDK ALONE, placed in this venv's site-packages.
Linking one module exposes the SDK without putting ~/.local back on the path, so
numpy and torch stay as the venv pinned them. Nothing is deleted or moved: the
original file stays exactly where it is, and removing the link undoes this
completely.

The SDK is a compiled C++ extension, so it only imports into the Python version
it was built against. This checks that before linking, because a silent ABI
mismatch is a much more confusing failure than a clear refusal.

    python tools/link_sdk.py            # find, check, link, verify
    python tools/link_sdk.py --dry-run  # say what it would do, change nothing
    python tools/link_sdk.py --unlink   # remove a link made earlier
"""

import argparse
import os
import site
import sys
import sysconfig
from pathlib import Path

MODULE = "booster_robotics_sdk_python"


def already_importable():
    """The path the module imports from right now, or None."""
    try:
        import importlib.util

        spec = importlib.util.find_spec(MODULE)
    except (ImportError, ValueError):
        return None
    if spec is None:
        return None
    return spec.origin or "(namespace package)"


def search_roots():
    """Directories worth searching, most likely first."""
    home = Path.home()
    roots = []

    # ~/.local for every Python version.
    for path in sorted(home.glob(".local/lib/python*/site-packages")):
        roots.append(path)

    # Other virtualenvs in the home directory. This is the common case on a
    # machine where one env was built for the robot SDK and another for the
    # vision stack: the SDK is sitting in a sibling env, not in a system path.
    # Same-version envs can share it, so searching here usually ends the hunt.
    for pattern in ("*/lib/python*/site-packages",
                    "*/*/lib/python*/site-packages"):
        for path in sorted(home.glob(pattern)):
            if path.is_dir():
                roots.append(path)

    # Conda / mamba environments, where the SDK may have been installed before
    # the venv existed.
    for base in (home / "miniconda3", home / "anaconda3", home / "miniforge3",
                 Path("/opt/conda")):
        for path in sorted(base.glob("envs/*/lib/python*/site-packages")):
            roots.append(path)
        for path in sorted(base.glob("lib/python*/site-packages")):
            roots.append(path)

    # System Python.
    for path in (Path("/usr/lib/python3/dist-packages"),
                 Path("/usr/local/lib/python3/dist-packages")):
        if path.is_dir():
            roots.append(path)
    for path in sorted(Path("/usr/local/lib").glob("python*/dist-packages")):
        roots.append(path)

    # A source build that was never installed: the Booster SDK is normally built
    # with cmake and left in the tree.
    for pattern in ("booster*", "*booster*/lib", "*booster*/build",
                    "*booster*/install"):
        for path in sorted(home.glob(pattern)):
            if path.is_dir():
                roots.append(path)

    seen = set()
    unique = []
    for path in roots:
        resolved = str(path.resolve())
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def find_candidates():
    """Every copy of the module on disk: (path, abi_tag_or_None)."""
    found = []
    seen = set()

    def consider(path):
        resolved = path.resolve()
        if resolved in seen:
            return
        seen.add(resolved)
        found.append((path, abi_tag(path)))

    for root in search_roots():
        if not root.is_dir():
            continue
        # A bounded walk: deep enough for build trees, shallow enough to be fast.
        for depth in ("", "*/", "*/*/", "*/*/*/"):
            for pattern in ("%s%s*.so" % (depth, MODULE),
                            "%s%s/__init__.py" % (depth, MODULE)):
                try:
                    for path in sorted(root.glob(pattern)):
                        consider(path)
                except OSError:
                    continue
    return found


def abi_tag(path):
    """The Python ABI a compiled module was built for, e.g. '310'. None if pure."""
    name = path.name
    if not name.endswith(".so"):
        return None
    for part in name.split("."):
        if part.startswith("cpython-"):
            digits = part[len("cpython-"):].split("-")[0]
            return digits if digits.isdigit() else None
    return None


def this_abi():
    return "%d%d" % (sys.version_info[0], sys.version_info[1])


def importable_target(path):
    """What to link: the module file, or its package directory."""
    if path.name == "__init__.py":
        return path.parent
    return path


def venv_site_packages():
    """This interpreter's own site-packages, or None if not in a venv."""
    if sys.prefix == getattr(sys, "base_prefix", sys.prefix):
        return None
    purelib = sysconfig.get_paths().get("purelib")
    if purelib and Path(purelib).is_dir():
        return Path(purelib)
    for path in site.getsitepackages():
        candidate = Path(path)
        if candidate.is_dir() and str(candidate).startswith(sys.prefix):
            return candidate
    return None


def verify():
    """Import the module in a FRESH interpreter, which is the real test."""
    import subprocess

    code = ("import %s as sdk; "
            "print(getattr(sdk, '__file__', 'built-in'))" % MODULE)
    result = subprocess.run([sys.executable, "-c", code],
                            capture_output=True, text=True)
    if result.returncode == 0:
        return True, result.stdout.strip()
    return False, (result.stderr.strip().splitlines() or ["(no error text)"])[-1]


def do_unlink(site_dir):
    removed = []
    for name in (MODULE, MODULE + ".so"):
        for path in sorted(site_dir.glob(name + "*")):
            if path.is_symlink():
                path.unlink()
                removed.append(path)
    if removed:
        for path in removed:
            print("  removed link %s" % path)
    else:
        print("  no links made by this script were found in %s" % site_dir)
    return bool(removed)


def main():
    parser = argparse.ArgumentParser(
        description="Make the Booster SDK importable from this interpreter")
    parser.add_argument("--dry-run", action="store_true",
                        help="report only; create nothing")
    parser.add_argument("--unlink", action="store_true",
                        help="remove a link made by an earlier run")
    args = parser.parse_args()

    print("interpreter: %s" % sys.executable)
    print("version:     %s (ABI cpython-%s)"
          % (sys.version.split()[0], this_abi()))
    site_dir = venv_site_packages()
    print("venv:        %s" % (sys.prefix if site_dir else "not in a venv"))
    if site_dir:
        print("site dir:    %s" % site_dir)
    print()

    if args.unlink:
        if site_dir is None:
            raise SystemExit("Not in a venv, so this script made no links.")
        do_unlink(site_dir)
        return

    origin = already_importable()
    if origin:
        print("Already importable from:\n  %s" % origin)
        print("\nNothing to do.")
        return

    print("%s is not importable here. Searching ..." % MODULE)
    candidates = find_candidates()
    if not candidates:
        print("\nNot found anywhere this script looked. The SDK is probably not")
        print("installed for any Python on this machine, or it lives somewhere")
        print("unusual. Find it with:")
        print("    sudo find / -name '%s*' -maxdepth 8 2>/dev/null" % MODULE)
        raise SystemExit(2)

    print("\nfound %d copy(ies):" % len(candidates))
    compatible = []
    for path, tag in candidates:
        if tag is None:
            note = "pure Python - works on any version"
            compatible.append(path)
        elif tag == this_abi():
            note = "built for cpython-%s - MATCHES this interpreter" % tag
            compatible.append(path)
        else:
            note = ("built for cpython-%s - will NOT import into cpython-%s"
                    % (tag, this_abi()))
        print("  %s\n      %s" % (path, note))

    if not compatible:
        print("\nEvery copy was built for a different Python version, so none can")
        print("be linked: a compiled extension cannot cross versions. Options:")
        print("  1. rebuild the SDK's Python binding against python%d.%d"
              % sys.version_info[:2])
        print("  2. recreate the venv on the Python the SDK was built for")
        print("Nothing was changed.")
        raise SystemExit(3)

    target = importable_target(compatible[0])
    if site_dir is None:
        print("\nNot in a venv, so there is nothing safe to link into. Instead:")
        print("    export PYTHONPATH=%s:$PYTHONPATH" % target.parent)
        raise SystemExit(4)

    link = site_dir / target.name
    print("\nlink to create:\n  %s\n    -> %s" % (link, target))

    if args.dry_run:
        print("\n--dry-run: nothing created.")
        return

    if link.exists() or link.is_symlink():
        if link.is_symlink() and link.resolve() == target.resolve():
            print("  that link already exists")
        else:
            raise SystemExit(
                "\n%s already exists and is not a link to the SDK.\n"
                "Refusing to touch it. Inspect it first." % link)
    else:
        os.symlink(target, link)
        print("  created")

    ok, detail = verify()
    print()
    if ok:
        print("VERIFIED: a fresh interpreter imports it from\n  %s" % detail)
        print("\nNow run:\n    python sdk_client/sdk_client.py walk")
    else:
        print("Link made, but importing still fails:\n  %s" % detail)
        print("\nThat usually means a missing shared library the module needs.")
        print("Check with:\n    ldd %s | grep 'not found'" % target)
        raise SystemExit(5)


if __name__ == "__main__":
    main()
