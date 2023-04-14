from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_people(file_name="names.csv"):
    """
    load people from names.csv
    """
    return pd.read_csv(file_name, sep=";")


def load_table_layout(file_name="table-layout.csv"):
    """
    Load table layout from csv
    """
    return pd.read_csv(file_name, sep=";").dropna()


def load_past_groupings(file_name: str | Path = "seating-plan.csv"):
    """
    Load seating plan from csv
    """
    return pd.read_csv(file_name, sep=";").dropna(how="all")
