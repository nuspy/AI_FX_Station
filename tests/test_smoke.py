def test_import_package():
    import importlib
def test_import_package():
    import importlib

    pkg = importlib.import_module("src.forex_diffusion")
    assert hasattr(pkg, "__version__")
    from src.forex_diffusion import utils, data, features, models, services, ui
    assert utils is not None
    assert data is not None
    assert features is not None
    assert models is not None
    assert services is not None
    assert ui is not None
    pkg = importlib.import_module("src.forex_diffusion")
    assert hasattr(pkg, "__version__")
    from src.forex_diffusion import utils, data, features, models, services, ui
    assert utils is not None
    assert data is not None
    assert features is not None
    assert models is not None
    assert services is not None
    assert ui is not None
