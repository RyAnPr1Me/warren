from src.utils.config_loader import load_config

def test_load_config():
    cfg = load_config()
    assert cfg.seq_len == 120
    assert cfg.batch_size == 256
