#
# create a zip file for submission
#

import os
import sys
import zipfile
import warnings

REQUIRED_FILES = {
    "run_llama.py",
    "llama.py",
    "optimizer.py",
    "classifier.py",
    "rope.py",
    "generated-sentence-temp-0.txt",
    "generated-sentence-temp-1.txt",
    "sst-dev-prompting-output.txt",
    "sst-test-prompting-output.txt",
    "cfimdb-dev-prompting-output.txt",
    "cfimdb-test-prompting-output.txt",
    "addition_data_generation.py",
    "addition_lib.py",
    "addition_run.py",
    "base_llama.py",
    "run_llama.py",
    "feedback.txt",
    "sanity_check.py",
    "sanity_check.data",
    "config.py",
    "setup.sh",
    "utils.py",
    "README.md",
    "structure.md",
    "checklist.md",
    "tokenizer.py",
}

REQUIRED_DIRS = {
    "addition_models",
}

OPTIONAL_FILES = {
    "feedback.txt",
}

REQUIRED_FILES = {os.path.normpath(f) for f in REQUIRED_FILES}
REQUIRED_DIRS = {os.path.normpath(d) for d in REQUIRED_DIRS}


def is_inside_required_dir(rel_root, required_dirs):
    for d in required_dirs:
        if rel_root == d or rel_root.startswith(d + os.sep):
            return True
    return False


def check_file(file: str, check_aid: str):
    missing_files = set(REQUIRED_FILES)
    missing_dirs = set(REQUIRED_DIRS)

    target_prefix = None
    inside_files = set()

    with zipfile.ZipFile(file, "r") as zz:
        print(f"Read zipfile {file}:")
        zz.printdir()
        print("#--")

        for info in zz.infolist():
            if info.filename.startswith("_"):
                continue

            if target_prefix is None:
                target_prefix, _ = info.filename.split("/", 1)
                target_prefix = target_prefix + "/"

            assert info.filename.startswith(
                target_prefix
            ), "There should only be one top-level dir (with your andrew id as the dir-name) inside the zip file."

            ff = info.filename[len(target_prefix) :].replace("\\", "/")
            inside_files.add(ff)

    # resolve files
    missing_files -= inside_files

    # resolve dirs
    for d in list(missing_dirs):
        if any(f.startswith(d + "/") for f in inside_files):
            missing_dirs.remove(d)

    combined_files = (missing_files | OPTIONAL_FILES) - inside_files

    assert len(missing_files) == 0, f"Some required files are missing: {missing_files}"
    assert (
        len(missing_dirs) == 0
    ), f"Some required directories are missing: {missing_dirs}"

    assert (
        target_prefix[:-1] == check_aid
    ), f"AndrewID mismatched: {target_prefix[:-1]} vs {check_aid}"

    print(
        f"Read zipfile {file}, please check that your andrew-id is: {target_prefix[:-1]}"
    )
    print(f"And it contains the following files: {sorted(list(inside_files))}")

    if len(combined_files) not in [0, 4]:
        warnings.warn(
            f"[Optional check] Some of your advanced outputs are missing: {combined_files}"
        )


def main(path: str, aid: str):
    aid = aid.strip()

    # local tracking only
    remaining_files = set(REQUIRED_FILES)
    remaining_dirs = set(REQUIRED_DIRS)

    if os.path.isdir(path):
        with zipfile.ZipFile(f"{aid}.zip", "w") as zz:
            for root, dirs, files in os.walk(path):
                if ".git" in root or "__pycache__" in root:
                    continue

                rel_root = os.path.normpath(os.path.relpath(root, path))
                inside_required_dir = is_inside_required_dir(rel_root, REQUIRED_DIRS)

                # DIRECTORY HANDLING
                if not inside_required_dir:
                    for d in list(dirs):
                        dir_rpath = os.path.normpath(os.path.join(rel_root, d))
                        if dir_rpath not in REQUIRED_DIRS:
                            dirs.remove(d)

                if inside_required_dir or rel_root in REQUIRED_DIRS:
                    zip_dir_path = os.path.join(aid, rel_root).replace("\\", "/") + "/"
                    zz.writestr(zip_dir_path, "")

                # FILE HANDLING
                for file in files:
                    if file.endswith(".zip"):
                        continue

                    rpath = os.path.normpath(os.path.join(rel_root, file))

                    if inside_required_dir or rpath in REQUIRED_FILES:
                        ff = os.path.join(root, file)
                        zip_path = os.path.join(aid, rpath).replace("\\", "/")
                        zz.write(ff, zip_path)

                        remaining_files.discard(rpath)
                        if inside_required_dir:
                            for d in list(remaining_dirs):
                                if rpath.startswith(d + os.sep):
                                    remaining_dirs.discard(d)

        print(
            f"required files are {remaining_files} and required dirs are {remaining_dirs}"
        )

        if remaining_files or remaining_dirs:
            print("Missing required files:", remaining_files)
            print("Missing required dirs:", remaining_dirs)
            breakpoint()

        print(f"Submission zip file created from DIR={path} for {aid}: {aid}.zip")
        check_file(f"{aid}.zip", aid)

    else:
        check_file(path, aid)


if __name__ == "__main__":
    main(*sys.argv[1:])
