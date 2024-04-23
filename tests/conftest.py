import pytest
from dotenv import load_dotenv


@pytest.fixture(autouse=True, scope="session")  # type: ignore
def env_file() -> None:
    """Load environment variables from env file."""
    load_dotenv()
