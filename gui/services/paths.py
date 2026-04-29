import os
from pathlib import Path


APP_NAME = "SpeakerRegQt"


def _appdata_root():
    root = os.environ.get("APPDATA")
    if root:
        return Path(root)
    return Path.home() / "AppData" / "Roaming"


def _localappdata_root():
    root = os.environ.get("LOCALAPPDATA")
    if root:
        return Path(root)
    return Path.home() / "AppData" / "Local"


def appdata_dir():
    path = _appdata_root() / APP_NAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def local_tmp_dir():
    path = _localappdata_root() / APP_NAME / "tmp"
    path.mkdir(parents=True, exist_ok=True)
    return path


def profiles_dir():
    path = appdata_dir() / "profiles"
    path.mkdir(parents=True, exist_ok=True)
    return path


def config_path():
    return appdata_dir() / "config.json"


def profiles_db_path():
    return appdata_dir() / "profiles.sqlite3"


def history_db_path():
    return appdata_dir() / "history.sqlite3"
