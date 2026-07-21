#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
data_root="${1:-${repo_root}/data/PubChem324kV2}"
archive="${data_root}/raw/PubChem324kV2.zip"
source_dir="${data_root}/source"
url="https://huggingface.co/datasets/acharkq/PubChem324kV2/resolve/e449660d39ec83c4ccf0bff2dcfb9bbf6943ab89/PubChem324kV2.zip?download=true"
expected_sha256="95a64b98b19deea22f30aee33a15a1dbf23dc14dbdcc62013c8923d2e3094f2d"

mkdir -p "${data_root}/raw" "${source_dir}"

if ! printf '%s  %s\n' "${expected_sha256}" "${archive}" | sha256sum --check --status; then
  temporary_archive="${archive}.partial"
  rm -f "${temporary_archive}"
  wget --output-document="${temporary_archive}" "${url}"
  printf '%s  %s\n' "${expected_sha256}" "${temporary_archive}" | sha256sum --check
  mv "${temporary_archive}" "${archive}"
fi

printf '%s  %s\n' "${expected_sha256}" "${archive}" | sha256sum --check

pretrain_file="${source_dir}/PubChem324kV2/pretrain.pt"
if [[ ! -s "${pretrain_file}" || "${archive}" -nt "${pretrain_file}" ]]; then
python3 - "${archive}" "${source_dir}" <<'PY'
from pathlib import Path
from zipfile import ZipFile
import sys

archive = Path(sys.argv[1])
destination = Path(sys.argv[2])
with ZipFile(archive) as zf:
    unsafe = [name for name in zf.namelist() if Path(name).is_absolute() or ".." in Path(name).parts]
    if unsafe:
        raise RuntimeError(f"Unsafe paths in archive: {unsafe[:3]}")
    zf.extractall(destination)
print(f"Extracted {archive} into {destination}")
PY
fi

find "${source_dir}" -maxdepth 3 -type f -name '*.pt' -print | sort
