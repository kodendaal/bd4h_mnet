import os, re, yaml

_env_pat = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\:([^}]*)\}")

def _expand_env_defaults(s: str) -> str:
    def repl(m):
        key, default = m.group(1), m.group(2)
        return os.environ.get(key, default)
    return _env_pat.sub(repl, s)

def load_yaml_with_env(path: str):
    with open(path, "r") as f:
        raw = f.read()
    raw = _expand_env_defaults(raw)
    return yaml.safe_load(raw)
