from ct_clf_hack.shared.file_utils import read_yaml
from ct_clf_hack.shared.path_utils import backend_configs_dpath

_example_config_path = backend_configs_dpath() / "db_cfg_example.yaml"
_example_config = read_yaml(_example_config_path)

_config_path = backend_configs_dpath() / "db_cfg.yaml"
_config = read_yaml(_config_path) if _config_path.exists() else _example_config

_db_config = _config.get("database", {})
DB_NAME = _db_config.get("name", "ct_clf")
DB_HOST = _db_config.get("host", "localhost")
DB_URL = _db_config.get("url", "sqlite:///ct_clf.db")
DB_PORT = _db_config.get("port", None)
