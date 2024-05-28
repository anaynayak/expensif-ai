import pytest
from pathlib import Path


@pytest.fixture(scope="module")
def llm_cache():
    import litellm
    from litellm.caching import Cache

    path = Path(__file__).parent / "images/.litellm_cache"
    litellm.cache = Cache(type="disk", disk_cache_dir=str(path.absolute()))
