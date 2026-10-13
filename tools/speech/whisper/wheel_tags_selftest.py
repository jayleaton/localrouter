"""venv/bin/python -I wheel_tags_selftest.py --wheels DIR --requirements FILE --report OUT.json

wheel_tags.py against the real locked wheel, in scratch site directories only (never the job's venv). First, before
anything is unpacked, the wheel is authenticated against its requirements sha256 pin and every archive member is
validated (wheel_tags.pinned_wheel, archive_members). Then real pip lays it out once into a scratch "pristine" site
(`python -I -m pip install --no-index --no-deps --require-hashes --target`, as the job installs, so RECORD has pip's
own form: sorted rows, \\r\\n line ends, hashed INSTALLER/REQUESTED, bytecode rows with an empty hash and size), with a
non-empty PYTHONPYCACHEPREFIX inherited on purpose: nothing may be written under it and every bytecode row must stay a
contained relative path (pip_layout in the report), or the self-test fails before any case. Each case runs on a fresh copy of that
site and the copy is removed as soon as its result is recorded; a cleanup failure is a failed case, not ignored.
  positive           as pip installed it                     -> 0; only WHEEL and the WHEEL line of RECORD change
  wrong_pin          requirements sha256 changed             -> 1
  metadata_changed   one byte of installed METADATA          -> 1
  library_changed    one byte of an installed shared library -> 1
  record_duplicate   a row of installed RECORD repeated      -> 1
  record_unsafe      installed RECORD row '../escape'        -> 1
  record_extra       installed RECORD row for an unlisted file -> 1
  bytecode_orphan    a bytecode row whose .py the wheel lacks -> 1
  bytecode_hashed    a pip bytecode row given a hash         -> 1 (not applicable if pip compiled nothing: reported)
  library_symlink    a shared library replaced by a symlink  -> 1
  unexpected_tag     the known bad tag set changed           -> 1
  unexpected_version the known version changed               -> 1
  already_corrected  run twice: the second refuses           -> 1
  interrupted        a failure injected after the WHEEL move -> 2, state wheel=corrected record=original
Every refused case must leave the site byte for byte as it was. Exit 1 if any case or any cleanup fails."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wheel_tags  # noqa: E402

K = wheel_tags.KNOWN
INFO = f"{K['dist']}-{K['version']}.dist-info"


def snapshot(site):
    out = {}
    for root, dirs, files in os.walk(site):
        for n in files + [d for d in dirs if os.path.islink(os.path.join(root, d))]:
            p = os.path.join(root, n)
            out[os.path.relpath(p, site)] = ("link:" + os.readlink(p)) if os.path.islink(p) \
                else hashlib.sha256(open(p, "rb").read()).hexdigest()
    return out


def record_path(site):
    return os.path.join(site, INFO, "RECORD")


def record_lines(site):
    return open(record_path(site), "rb").read().decode("utf-8").splitlines(keepends=True)


def ending(site):
    first = record_lines(site)[0]
    return first[len(first.rstrip("\r\n")):]


def append_record(site, line):
    with open(record_path(site), "ab") as f:
        f.write((line + ending(site)).encode("utf-8"))


def replace_record_line(site, prefix, new):
    lines = record_lines(site)
    i = next(i for i, ln in enumerate(lines) if ln.startswith(prefix))
    lines[i] = new + lines[i][len(lines[i].rstrip("\r\n")):]
    open(record_path(site), "wb").write("".join(lines).encode("utf-8"))


def first_library(site):
    for ln in record_lines(site):
        path = ln.split(",")[0]
        if ".so" in path.rsplit("/", 1)[-1]:
            return os.path.join(site, *path.split("/"))
    raise RuntimeError("no shared library in RECORD")


def bytecode_rows(site):
    return [ln.split(",")[0] for ln in record_lines(site) if wheel_tags.PYC.fullmatch(ln.split(",")[0])]


def flip_byte(path):
    data = bytearray(open(path, "rb").read())
    data[-1] ^= 0xFF
    open(path, "wb").write(bytes(data))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wheels", required=True)
    ap.add_argument("--requirements", required=True)
    ap.add_argument("--report", required=True)
    a = ap.parse_args()
    results, failed, root = [], 0, None
    report = {"failures": None, "cases": results, "cleanup_errors": []}

    def done(name, ok, **detail):
        nonlocal failed
        results.append(dict(case=name, ok=ok, **detail))
        failed += not ok
        print(f"{'PASS' if ok else 'FAIL'} {name} {json.dumps(detail)[:300]}", flush=True)

    try:
        # Authenticate the pinned wheel and validate every member before anything is unpacked.
        wheel_path, wheel_bytes, pin = wheel_tags.pinned_wheel(a.wheels, a.requirements)
        wheel_tags.archive_members(wheel_bytes)
        root = tempfile.mkdtemp(prefix="wheel-tags-selftest-")
        req1 = os.path.join(root, "requirements-one.txt")
        with open(req1, "w", encoding="utf-8") as f:
            f.write(f"{K['project']}=={K['version']} --hash=sha256:{pin}\n")
        pristine = os.path.join(root, "pristine")
        # Laid out exactly as the job installs (pip -I), with a non-empty PYTHONPYCACHEPREFIX inherited on purpose:
        # the bytecode must still land beside its source in the site, and nothing may be written to that prefix.
        outside = os.path.join(root, "inherited-pycache-prefix")
        env = dict(os.environ, PYTHONPYCACHEPREFIX=outside)
        r = subprocess.run([sys.executable, "-I", "-m", "pip", "install", "--no-index", "--no-deps", "--require-hashes",
                            "--find-links", a.wheels, "--target", pristine, "-r", req1],
                           capture_output=True, text=True, env=env)
        if r.returncode != 0:
            raise RuntimeError(f"pip could not lay out the wheel: {r.stdout[-500:]} {r.stderr[-500:]}")
        written = [os.path.relpath(os.path.join(d, f), outside) for d, _, fs in os.walk(outside) for f in fs]
        rows = bytecode_rows(pristine)
        report["pip_layout"] = {"record_line_ending": repr(ending(pristine)), "bytecode_rows": len(rows),
                                "inherited_pycache_prefix": outside, "writes_under_prefix": written[:20],
                                "bytecode_rows_contained": all(wheel_tags.safe_rel(p) for p in rows)}
        if written or not report["pip_layout"]["bytecode_rows_contained"]:
            raise RuntimeError(f"pip -I wrote bytecode outside the site: {report['pip_layout']}")
        bad_req = os.path.join(root, "requirements-wrong-pin.txt")
        with open(bad_req, "w", encoding="utf-8") as f:
            f.write(f"{K['project']}=={K['version']} --hash=sha256:{('0' if pin[0] != '0' else '1') + pin[1:]}\n")

        def case(name, expect, mutate=None, known=K, requirements=req1, fault=None, twice=False):
            work = os.path.join(root, name)
            site = os.path.join(work, "site")
            try:
                shutil.copytree(pristine, site, symlinks=True)
                if mutate:
                    mutate(site)
                if twice:
                    wheel_tags.run(site, a.wheels, requirements, os.path.join(work, "audit-first"), known)
                before = snapshot(site)
                code, outcome = wheel_tags.run(site, a.wheels, requirements, os.path.join(work, "audit"), known, fault)
                after = snapshot(site)
                ok = code == expect
                detail = {"expect": expect, "code": code, "outcome": outcome.get("outcome"),
                          "reason": (outcome.get("reason") or "")[:200]}
                if expect == 1:
                    ok &= before == after
                    detail["unchanged"] = before == after
                elif expect == 0:
                    changed = sorted(k for k in after if before.get(k) != after[k])
                    lines_b = [ln for ln in record_lines(pristine)]
                    lines_a = record_lines(site)
                    diff = [i for i, (x, y) in enumerate(zip(lines_b, lines_a)) if x != y]
                    tags = wheel_tags.tags_of(open(os.path.join(site, INFO, "WHEEL"), "rb").read())
                    ok &= changed == sorted([f"{INFO}/WHEEL", f"{INFO}/RECORD"]) and len(lines_b) == len(lines_a) \
                        and len(diff) == 1 and lines_a[diff[0]].startswith(f"{INFO}/WHEEL,") and tags == {K["good_tag"]}
                    detail.update(changed_files=changed, record_lines_changed=len(diff), tags_after=sorted(tags),
                                  bytecode_rows=len(outcome.get("bytecode_rows", [])),
                                  libraries=len(outcome.get("native_libraries", [])))
                else:
                    ok &= outcome.get("state") == {"wheel": "corrected", "record": "original"}
                    detail["state"] = outcome.get("state")
            except Exception as e:  # a case that cannot be set up or run is a failure, never a pass
                ok, detail = False, {"expect": expect, "error": repr(e)[:300]}
            finally:
                try:
                    shutil.rmtree(work)
                except OSError as e:
                    ok = False
                    report["cleanup_errors"].append(f"{name}: {e}")
            done(name, ok, **detail)

        def symlink_library(site):
            lib = first_library(site)
            shutil.copyfile(lib, lib + ".real")
            os.remove(lib)
            os.symlink(os.path.basename(lib) + ".real", lib)

        def duplicate_row(site):
            line = next(ln for ln in record_lines(site) if ln.startswith(f"{INFO}/METADATA,"))
            append_record(site, line.rstrip("\r\n"))

        def extra_row(site):
            data = b"extra\n"
            open(os.path.join(site, INFO, "EXTRA"), "wb").write(data)
            append_record(site, f"{INFO}/EXTRA,{wheel_tags.record_hash(data)},{len(data)}")

        def orphan_bytecode(site):
            rel = f"{INFO.split('-')[0]}_selftest/__pycache__/absent.cpython-312.pyc"
            os.makedirs(os.path.join(site, os.path.dirname(rel)))
            open(os.path.join(site, rel), "wb").write(b"\0" * 16)
            append_record(site, f"{rel},,")

        def hashed_bytecode(site):
            rel = bytecode_rows(site)[0]
            data = open(os.path.join(site, rel), "rb").read()
            replace_record_line(site, rel + ",", f"{rel},{wheel_tags.record_hash(data)},{len(data)}")

        case("positive", 0)
        case("wrong_pin", 1, requirements=bad_req)
        case("metadata_changed", 1, lambda s: flip_byte(os.path.join(s, INFO, "METADATA")))
        case("library_changed", 1, lambda s: flip_byte(first_library(s)))
        case("record_duplicate", 1, duplicate_row)
        case("record_unsafe", 1, lambda s: append_record(s, "../escape,sha256=AAAA,1"))
        case("record_extra", 1, extra_row)
        case("bytecode_orphan", 1, orphan_bytecode)
        if bytecode_rows(pristine):
            case("bytecode_hashed", 1, hashed_bytecode)
        else:
            results.append({"case": "bytecode_hashed", "ok": True, "not_applicable": "pip compiled no .py of this wheel"})
        case("library_symlink", 1, symlink_library)
        case("unexpected_tag", 1, known=dict(K, bad_tags={"py3-none-manylinux2014_x86_64"}))
        case("unexpected_version", 1, known=dict(K, version="0.8.0"))
        case("already_corrected", 1, twice=True)
        case("interrupted", 2, fault="after_wheel")
    except Exception as e:  # authentication, validation or layout failed: nothing ran, the self-test fails
        done("setup", False, error=repr(e)[:500])
    finally:
        if root:
            try:
                shutil.rmtree(root)
            except OSError as e:
                report["cleanup_errors"].append(f"root: {e}")
    failed += len([e for e in report["cleanup_errors"] if e.startswith("root:")])
    report["failures"] = failed
    os.makedirs(os.path.dirname(os.path.abspath(a.report)), exist_ok=True)
    with open(a.report, "w") as f:
        json.dump(report, f, indent=1)
    print(f"{len(results) - failed}/{len(results)} cases passed; cleanup errors {len(report['cleanup_errors'])}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
