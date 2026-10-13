"""venv/bin/python -I wheel_tags.py --site SITE --wheels DIR --requirements FILE --audit DIR

Corrects one proven upstream label defect in installed wheel metadata so an unmodified `pip check` can judge the
environment. Run after `pip install --require-hashes` and before `pip check`.
  exit 0  repaired: proven, written, read back, and the audit record persisted
  exit 1  refused: nothing in SITE was changed (anything not exactly as below refuses, an already corrected WHEEL and
          any read, parse or I/O error before the first move included)
  exit 2  UNPROVEN: something changed or may have changed and the result could not be proven (a failure after the
          first move, a read-back that differs, or an audit record that could not be persisted after a repair); the
          environment must not be used
The outcome is written to AUDIT/outcome.json and printed.

The defect, for the one wheel this project locks (KNOWN below): the file is named
nvidia_cusparselt_cu13-0.8.1-py3-none-manylinux2014_aarch64.whl, so `pip install` accepts it on aarch64, but if its
own .dist-info/WHEEL declares `Tag: py3-none-manylinux2014_sbsa` (a platform tag no installer supports), `pip check`
reports "is not supported on this platform" (pip 25.0.1, operations/check.py check_unsupported: the WHEEL file's Tag
set must intersect the interpreter's tags). The B2 v6 run on GB10 failed exactly that pip check.

Nothing is changed unless all of this is proven first:
  1. requirements.txt pins the distribution at exactly KNOWN's version with exactly one sha256, and exactly one wheel
     in WHEELS has that name and version, the corrected tag in its file name, and that sha256;
  2. the wheel archive is sound: member names are relative, without '..', backslashes, NUL, duplicates or .data dirs,
     and none is a symlink or special file; its RECORD parses as strict CSV (3 fields a row, unique safe paths), lists
     exactly the archive's files, and every row's sha256 and size equal the member's bytes;
  3. the archive's WHEEL Tag set is exactly KNOWN's bad set, and its METADATA names the distribution and version;
  4. the installed copy is that archive, as pip 25.0.1 lays it out (operations/install/wheel.py
     get_csv_rows_for_installed): the installed WHEEL and METADATA bytes equal the archive's; the installed RECORD
     parses strictly and holds the archive's rows unchanged, plus only pip's own: INSTALLER and REQUESTED (hashed by
     pip; verified here) and bytecode of the archive's .py files (rows with an empty hash and size, as pip writes
     them: the file must exist, regular, inside SITE, and its digest is computed and recorded in the audit, not
     invented as an original); every archive file is a regular file (no symlink in its path) inside SITE whose bytes
     match its row; every shared library is a 64-bit little-endian AArch64 ELF.
Then the original WHEEL and RECORD bytes are copied to AUDIT; the new WHEEL (only Tag lines changed) and the new
RECORD (only the WHEEL row's line replaced, its line ending kept: pip writes RECORD with csv's \\r\\n) are written to
fresh exclusive temporaries in the same directories, fsynced, and moved into place WHEEL first, RECORD second. Both
are read back and checked."""
import argparse
import base64
import csv
import email.parser
import hashlib
import io
import json
import os
import re
import stat
import sys
import tempfile
import zipfile

from packaging.tags import parse_tag, sys_tags

# The locked wheel only. Another version needs its own evidence before it is added.
KNOWN = {
    "dist": "nvidia_cusparselt_cu13", "project": "nvidia-cusparselt-cu13", "version": "0.8.1",
    "bad_tags": {"py3-none-manylinux2014_sbsa"}, "good_tag": "py3-none-manylinux2014_aarch64",
}
EM_AARCH64 = 183
PIP_HASHED = {"INSTALLER", "REQUESTED"}
PYC = re.compile(r"((?:[^/]+/)*)__pycache__/([^/]+)\.cpython-\d+(?:\.opt-[12])?\.pyc")
READ_ERRORS = (OSError, UnicodeDecodeError, csv.Error, zipfile.BadZipFile, ValueError, KeyError, IndexError)


class Refuse(Exception):
    """Not proven: nothing was changed."""


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def record_hash(data):
    return "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()


def safe_rel(path):
    return bool(path) and not path.startswith("/") and "\\" not in path and "\0" not in path \
        and all(part not in ("", ".", "..") for part in path.split("/"))


def parse_record(text, where):
    """Strict RECORD: 3 CSV fields a row, unique safe paths; returns {path: (hash, size or None)}."""
    rows = {}
    for row in csv.reader(io.StringIO(text, newline=""), strict=True):
        if len(row) != 3:
            raise Refuse(f"{where}: RECORD row with {len(row)} fields: {row!r}")
        path, h, size = row
        if not safe_rel(path):
            raise Refuse(f"{where}: unsafe RECORD path {path!r}")
        if path in rows:
            raise Refuse(f"{where}: duplicate RECORD path {path!r}")
        if (h == "") != (size == "") or (size and not size.isdigit()) or (h and not h.startswith("sha256=")):
            raise Refuse(f"{where}: malformed RECORD row {row!r}")
        rows[path] = (h, int(size) if size else None)
    return rows


def pinned_hash(requirements, project, version):
    pins = []
    for line in open(requirements, encoding="utf-8").read().replace("\\\n", " ").splitlines():
        parts = line.split()
        if parts and parts[0].lower() == f"{project}=={version}":
            pins.append(re.findall(r"--hash=sha256:([0-9a-f]{64})", line))
    if len(pins) != 1 or len(pins[0]) != 1:
        raise Refuse(f"requirements pin {project}=={version} must appear once with exactly one sha256, found {pins}")
    return pins[0][0]


def tags_of(wheel_bytes):
    return set(email.parser.Parser().parsestr(wheel_bytes.decode("utf-8")).get_all("Tag", []))


def pinned_wheel(wheels, requirements, known=KNOWN):
    """(path, bytes, sha256) of the one wheel file the requirements pin; Refuse otherwise."""
    name, version, good = known["dist"], known["version"], known["good_tag"]
    want = pinned_hash(requirements, known["project"], version)
    files = [f for f in os.listdir(wheels) if f.startswith(f"{name}-{version}-") and f.endswith(".whl")]
    if files != [f"{name}-{version}-{good}.whl"]:
        raise Refuse(f"expected exactly {name}-{version}-{good}.whl in {wheels}, found {files}")
    path = os.path.join(wheels, files[0])
    if not stat.S_ISREG(os.lstat(path).st_mode):
        raise Refuse(f"{files[0]} is not a regular file")
    data = open(path, "rb").read()
    if sha256(data) != want:
        raise Refuse(f"{files[0]}: sha256 {sha256(data)} is not the requirements pin {want}")
    return path, data, want


def archive_members(wheel_bytes, known=KNOWN):
    """{name: bytes} of a sound archive whose RECORD verifies every member (rules 2 and 3); Refuse otherwise."""
    info = f"{known['dist']}-{known['version']}.dist-info"
    members = {}
    with zipfile.ZipFile(io.BytesIO(wheel_bytes)) as z:
        for zi in z.infolist():
            kind = (zi.external_attr >> 16) & 0o170000
            if zi.is_dir():
                if kind not in (0, stat.S_IFDIR) or not safe_rel(zi.filename.rstrip("/")):
                    raise Refuse(f"archive directory {zi.filename!r} is unsafe")
                continue
            if kind not in (0, stat.S_IFREG):
                raise Refuse(f"archive member {zi.filename!r} is a symlink or special file")
            if not safe_rel(zi.filename) or zi.filename in members or ".data/" in zi.filename:
                raise Refuse(f"archive member {zi.filename!r} is unsafe, duplicated or a .data entry")
            members[zi.filename] = z.read(zi)
    rec_name = f"{info}/RECORD"
    if rec_name not in members:
        raise Refuse("the archive has no RECORD")
    record = parse_record(members[rec_name].decode("utf-8"), "archive")
    if set(record) != set(members):
        raise Refuse(f"archive RECORD and members differ: {sorted(set(record) ^ set(members))[:10]}")
    for path, (h, size) in record.items():
        if path == rec_name:
            if h or size is not None:
                raise Refuse("archive RECORD lists a hash for itself")
        elif h != record_hash(members[path]) or size != len(members[path]):
            raise Refuse(f"archive member {path} does not match its RECORD row")
    wheel, meta = members.get(f"{info}/WHEEL"), members.get(f"{info}/METADATA")
    if wheel is None or meta is None:
        raise Refuse("the archive lacks WHEEL or METADATA")
    if tags_of(wheel) != known["bad_tags"]:
        raise Refuse(f"archive WHEEL tags are {sorted(tags_of(wheel))}, not the known defect {sorted(known['bad_tags'])}")
    m = email.parser.Parser().parsestr(meta.decode("utf-8"))
    if (m["Name"] or "").replace("-", "_").lower() != known["dist"] or m["Version"] != known["version"]:
        raise Refuse(f"archive METADATA is {m['Name']} {m['Version']}")
    return members, record


def regular_inside(site, rel):
    """The installed path of `rel` if it is a regular file with no symlink anywhere below SITE; else Refuse."""
    path = site
    for part in rel.split("/"):
        path = os.path.join(path, part)
        if os.path.islink(path):
            raise Refuse(f"{rel}: symlink in the installed path")
    if not os.path.isfile(path) or os.path.realpath(path) != os.path.join(os.path.realpath(site), *rel.split("/")):
        raise Refuse(f"{rel}: not a regular file inside the site directory")
    return path


def aarch64_elf(data):
    return len(data) >= 20 and data[:4] == b"\x7fELF" and data[4] == 2 and data[5] == 1 \
        and int.from_bytes(data[18:20], "little") == EM_AARCH64


def prove(site, wheels, requirements, known=KNOWN):
    """Everything up to the write; returns the plan, or raises Refuse. Reads only."""
    name, version, good = known["dist"], known["version"], known["good_tag"]
    info = f"{name}-{version}.dist-info"
    if not os.path.isdir(os.path.join(site, info)):
        raise Refuse(f"{info} is not installed")
    wheel_file, wheel_bytes, pin = pinned_wheel(wheels, requirements, known)
    if not parse_tag(good) & set(sys_tags()):
        raise Refuse(f"{good} is not a supported tag of this interpreter")
    members, arc_record = archive_members(wheel_bytes, known)
    arc_wheel, rec_name = members[f"{info}/WHEEL"], f"{info}/RECORD"

    if open(regular_inside(site, f"{info}/WHEEL"), "rb").read() != arc_wheel \
            or open(regular_inside(site, f"{info}/METADATA"), "rb").read() != members[f"{info}/METADATA"]:
        raise Refuse("installed WHEEL or METADATA differs from the archive")
    record_bytes = open(regular_inside(site, rec_name), "rb").read()
    inst_record = parse_record(record_bytes.decode("utf-8"), "installed")
    missing = set(arc_record) - set(inst_record)
    if missing:
        raise Refuse(f"installed RECORD lacks archive rows: {sorted(missing)[:10]}")
    py_sources = {p for p in arc_record if p.endswith(".py")}
    pip_rows, bytecode = [], []
    for path, row in inst_record.items():
        if path in arc_record:
            if row != arc_record[path]:
                raise Refuse(f"installed RECORD row for {path} differs from the archive")
            continue
        base = path.rsplit("/", 1)[-1]
        m = PYC.fullmatch(path)
        if path == f"{info}/{base}" and base in PIP_HASHED:
            data = open(regular_inside(site, path), "rb").read()
            if row != (record_hash(data), len(data)):
                raise Refuse(f"installed {path} does not match its RECORD row")
            pip_rows.append(path)
        elif m and f"{m.group(1)}{m.group(2)}.py" in py_sources:
            if row != ("", None):  # pip 25.0.1 records compiled bytecode with an empty hash and size
                raise Refuse(f"bytecode row {path!r} is not in pip's form (empty hash and size): {row}")
            data = open(regular_inside(site, path), "rb").read()
            bytecode.append({"path": path, "computed_sha256": sha256(data), "size": len(data)})
        else:
            raise Refuse(f"installed RECORD has an unexpected row {path!r} {row}")
    libraries = []
    for path, (h, size) in arc_record.items():
        if path == rec_name:
            continue
        data = open(regular_inside(site, path), "rb").read()
        if (record_hash(data), len(data)) != (h, size):
            raise Refuse(f"installed {path} differs from the archive")
        if ".so" in path.rsplit("/", 1)[-1]:
            if not aarch64_elf(data):
                raise Refuse(f"{path} is not a 64-bit little-endian AArch64 ELF")
            libraries.append({"path": path, "sha256": sha256(data), "size": len(data)})
    if not libraries:
        raise Refuse("the wheel has no shared library")

    new_wheel = "".join(f"Tag: {good}\n" if ln.startswith("Tag:") else ln
                        for ln in arc_wheel.decode("utf-8").splitlines(keepends=True)).encode("utf-8")
    if tags_of(new_wheel) != {good}:
        raise Refuse("the corrected WHEEL does not parse to the corrected tag")
    # Replace only the WHEEL row's line, keeping its line ending and every other byte of RECORD.
    wheel_row = f"{info}/WHEEL"
    lines = record_bytes.decode("utf-8").splitlines(keepends=True)
    hits = [i for i, ln in enumerate(lines) if next(csv.reader([ln.rstrip("\r\n")], strict=True))[0] == wheel_row]
    if len(hits) != 1:
        raise Refuse(f"installed RECORD has {len(hits)} lines for {wheel_row}")
    old_line = lines[hits[0]]
    ending = old_line[len(old_line.rstrip("\r\n")):]
    buf = io.StringIO()
    csv.writer(buf, lineterminator=ending).writerow([wheel_row, record_hash(new_wheel), str(len(new_wheel))])
    lines[hits[0]] = buf.getvalue()
    new_record = "".join(lines).encode("utf-8")
    after = parse_record(new_record.decode("utf-8"), "corrected")
    if set(after) != set(inst_record) or any(after[p] != inst_record[p] for p in after if p != wheel_row):
        raise Refuse("the corrected RECORD changes more than the WHEEL row")
    return {
        "dist": f"{known['project']}=={version}", "wheel_file": wheel_file, "wheel_sha256": pin,
        "archive_record_rows": len(arc_record), "installed_record_rows": len(inst_record),
        "verified_installed_files": len(arc_record) - 1, "pip_hashed_rows": pip_rows, "bytecode_rows": bytecode,
        "native_libraries": libraries, "tags_before": sorted(tags_of(arc_wheel)), "tag_after": good,
        "changed_rows": [{"before": old_line.rstrip("\r\n"), "after": lines[hits[0]].rstrip("\r\n")}],
        "paths": {"wheel": os.path.join(site, wheel_row), "record": os.path.join(site, rec_name)},
        "bytes": {"wheel_before": arc_wheel, "wheel_after": new_wheel,
                  "record_before": record_bytes, "record_after": new_record},
    }


def write_pair(plan, audit, fault=None):
    """Audit copies, exclusive fsynced temporaries, then WHEEL and RECORD moved in that order and read back.
    Returns (outcome, state, reason); 'refused' only while nothing in SITE has moved."""
    b, p = plan["bytes"], plan["paths"]
    state = {"wheel": "original", "record": "original"}
    tmps = []
    try:
        os.makedirs(audit, exist_ok=True)
        for key in ("wheel_before", "record_before"):
            with open(os.path.join(audit, key), "xb") as f:
                f.write(b[key])
        moves = []
        for key, target in (("wheel_after", p["wheel"]), ("record_after", p["record"])):
            fd, tmp = tempfile.mkstemp(dir=os.path.dirname(target), prefix=".wheel_tags-")  # O_EXCL, fresh name
            tmps.append(tmp)
            with os.fdopen(fd, "wb") as f:
                f.write(b[key])
                f.flush()
                os.fsync(f.fileno())
            moves.append((tmp, target, key))
    except OSError as e:
        return "refused", state, f"before any move: {e}", tmps
    try:
        for tmp, target, key in moves:
            if fault == "after_wheel" and key == "record_after":
                raise OSError("injected failure after the WHEEL move")
            os.replace(tmp, target)
            state["wheel" if key == "wheel_after" else "record"] = "corrected"
        if open(p["wheel"], "rb").read() != b["wheel_after"] or open(p["record"], "rb").read() != b["record_after"]:
            return "unproven", state, "read-back differs from what was written", tmps
    except OSError as e:
        if state["wheel"] == "original":
            return "refused", state, f"the WHEEL move failed, nothing moved: {e}", tmps
        return "unproven", state, f"after the WHEEL move: {e}", tmps
    return "repaired", state, "", tmps


def run(site, wheels, requirements, audit, known=KNOWN, fault=None):
    """The whole step; returns (exit code, outcome dict) and persists AUDIT/outcome.json."""
    try:
        plan = prove(site, wheels, requirements, known)
    except Refuse as e:
        outcome = {"outcome": "refused", "reason": str(e)}
    except READ_ERRORS as e:  # prove() only reads: nothing changed
        outcome = {"outcome": "refused", "reason": f"{type(e).__name__} while proving: {e}"}
    else:
        b = plan.pop("bytes")
        try:
            result, state, reason, tmps = write_pair(dict(plan, bytes=b), audit, fault)
        except BaseException as e:  # unexpected: whatever moved is unknown
            result, state, reason, tmps = "unproven", {"wheel": "unknown", "record": "unknown"}, repr(e), []
        leftovers = []
        for tmp in tmps:
            try:
                if os.path.lexists(tmp):
                    os.remove(tmp)
            except OSError as e:
                leftovers.append(f"{tmp}: {e}")
        plan.pop("paths")
        outcome = dict(plan, outcome=result, state=state, reason=reason, temporaries_left=leftovers,
                       bytecode_policy={"pip": "python -I -m pip (PYTHON* ignored)",
                                        "inherited_PYTHONPYCACHEPREFIX": os.environ.get("PYTHONPYCACHEPREFIX"),
                                        "this_process_pycache_prefix": sys.pycache_prefix},
                       wheel_before_sha256=sha256(b["wheel_before"]), wheel_after_sha256=sha256(b["wheel_after"]),
                       record_before_sha256=sha256(b["record_before"]), record_after_sha256=sha256(b["record_after"]))
    code = {"repaired": 0, "refused": 1}.get(outcome["outcome"], 2)
    try:
        os.makedirs(audit, exist_ok=True)
        with open(os.path.join(audit, "outcome.json"), "w", encoding="utf-8") as f:
            json.dump(outcome, f, indent=1)
            f.flush()
            os.fsync(f.fileno())
    except OSError as e:
        outcome["audit_error"] = str(e)
        if code == 0:  # a repair whose record cannot be kept is not a proven repair
            outcome["outcome"], code = "unproven", 2
    return code, outcome


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site", required=True)
    ap.add_argument("--wheels", required=True)
    ap.add_argument("--requirements", required=True)
    ap.add_argument("--audit", required=True)
    a = ap.parse_args()
    code, outcome = run(os.path.abspath(a.site), a.wheels, a.requirements, a.audit)
    print("wheel_tags: " + json.dumps(outcome), flush=True)
    sys.exit(code)


if __name__ == "__main__":
    main()
