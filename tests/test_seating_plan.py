"""File to test the seating plan methods using PyTest
"""
import logging
import pytest

import pandas as pd

from beachbums.persons import Person
from beachbums.seating_plan import create_seating_plan
from beachbums.__main__ import load_people, load_table_layout, load_past_groupings

logger = logging.getLogger(__name__)


@pytest.fixture
def people() -> pd.DataFrame:
    """
    Load people from names.csv
    """
    return load_people(file_name="data/test_names.csv")
