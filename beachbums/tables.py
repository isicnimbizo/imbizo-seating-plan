import logging

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

from .persons import Person

logger = logging.getLogger(__name__)

@dataclass
class Table:
    """
    Table class
    """

    name: str
    size: int
    people: List[Person] = field(default_factory=lambda: [])
    actual_size: int = 0

    def add_person(self, person: Person):
        """
        Add a person to the table
        """
        if self.actual_size >= self.size:
            raise ValueError(f"Table {self.name} is full")
        if isinstance(person, str):
            logger.warning(f"{person} is not a Person object")
        self.people.append(person)
        self.actual_size += 1
        return self.actual_size

    def remove_person(self, person: Person):
        """
        Remove a person from the table
        """
        self.people.remove(person)
        self.actual_size -= 1
        return self.actual_size

    def __repr__(self):
        return f"{self.name} ({self.actual_size}/{self.size})"

    def __str__(self):
        return f"{self.name} = {[p.name for p in self.people]}"


def define_table_layout(layout: pd.DataFrame):
    """
    Define the table layout
    """
    logger.info("Defining table layout")
    tables = {}
    # change layout column names to uppercase
    layout.columns = [col.upper() for col in layout.columns]
    for table in layout.to_dict("records"):
        name: str = str(table["TABLE"]).lower()
        size = int(table["SIZE"])
        if "table" not in name:
            name = f"table {name}"
        name = name.capitalize()
        if name in tables:
            raise ValueError(f"Table {name} already defined")
        tables[name] = Table(name, size)
    return tables


def create_random_tables(
    num_tables=6, min_total_size=40, min_max_variation=2
) -> Dict[str, dict]:
    """
    Create a dict of dicts of random tables
    """
    tables = {}
    min_size = min_total_size // num_tables + 1
    max_size = min_size + min_max_variation
    for i in range(1, num_tables + 1):
        name = f"table {i}"
        tables[name] = Table(name, np.random.randint(min_size, max_size))
    return tables
