import io
import os
import tarfile
import tempfile
from typing import Tuple


def create_tarball() -> Tuple[str, tarfile.TarFile]:
    """Create an empty gzipped tarball in /tmp with owner-only permissions (0o700).

    Returns (path, tarfile_handle). The caller must close the handle when done.
    """
    fd = os.open(
        tempfile.mktemp(suffix=".tar.gz", dir="/tmp"),
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o700,
    )
    path = os.readlink(f"/proc/self/fd/{fd}")
    fileobj = os.fdopen(fd, "wb")
    tf = tarfile.open(fileobj=fileobj, mode="w:gz")
    tf._vast_path = path
    tf._vast_fileobj = fileobj
    return path, tf


def _sanitize_info(info: tarfile.TarInfo) -> tarfile.TarInfo:
    """Set uid/gid to 0 on a TarInfo."""
    info.uid = 0
    info.gid = 0
    info.uname = "root"
    info.gname = "root"
    return info


def add_file(tf: tarfile.TarFile, src_path: str, arcname: str) -> None:
    """Add a single file to the tarball under arcname, with uid/gid 0."""
    info = tf.gettarinfo(src_path, arcname=arcname)
    _sanitize_info(info)
    with open(src_path, "rb") as f:
        tf.addfile(info, f)


def add_folder(tf: tarfile.TarFile, src_path: str, arcname: str) -> None:
    """Recursively add a folder to the tarball, renaming the root to arcname, with uid/gid 0."""
    src_path = os.path.normpath(src_path)
    for root, dirs, files in os.walk(src_path):
        rel = os.path.relpath(root, src_path)
        arc_root = arcname if rel == "." else os.path.join(arcname, rel)

        # Add directory entry
        info = tf.gettarinfo(root, arcname=arc_root)
        _sanitize_info(info)
        tf.addfile(info)

        for fname in files:
            file_path = os.path.join(root, fname)
            arc_path = os.path.join(arc_root, fname)
            info = tf.gettarinfo(file_path, arcname=arc_path)
            _sanitize_info(info)
            with open(file_path, "rb") as f:
                tf.addfile(info, f)


def add_string(
    tf: tarfile.TarFile,
    content: str,
    arcname: str,
    executable: bool = False,
) -> None:
    """Add a string as a text file to the tarball with uid/gid 0.

    If executable=True, the file gets mode 0o755; otherwise 0o644.
    """
    data = content.encode("utf-8")
    info = tarfile.TarInfo(name=arcname)
    info.size = len(data)
    info.mode = 0o755 if executable else 0o644
    _sanitize_info(info)
    tf.addfile(info, io.BytesIO(data))
