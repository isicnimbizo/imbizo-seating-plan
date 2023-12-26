from __future__ import annotations

import logging
from dataclasses import dataclass, field

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
    capacity: int
    people: list[Person] = field(default_factory=lambda: [])

    @property
    def seated(self):
        return len(self.people)

    @property
    def has_space(self):
        return self.seated < self.capacity

    @property
    def is_full(self):
        return not self.has_space

    def add_person(self, person: Person):
        """
        Add a person to the table
        """
        if self.seated >= self.capacity:
            raise ValueError(f"Table {self.name} is full")
        if isinstance(person, str):
            logger.warning(f"{person} is not a Person object")
        self.people.append(person)
        return self.seated

    def remove_person(self, person: Person):
        """
        Remove a person from the table
        """
        self.people.remove(person)
        return self.seated

    def __repr__(self):
        return f"{self.name}({self.seated}/{self.capacity}) = {[p.name for p in self.people]}"

    def __str__(self):
        return f"{self.name}({self.seated}/{self.capacity}) = {[p.name for p in self.people]}"


def define_table_layout(layout: pd.DataFrame) -> dict[str, Table]:
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
) -> dict[str, Table]:
    """
    Create a dict of random tables
    """
    tables = {}
    min_size = min_total_size // num_tables + 1
    max_size = min_size + max(1, min_max_variation)  # +1 to avoid min_size == max_size
    for i in range(1, num_tables + 1):
        name = f"table {i}"
        tables[name] = Table(name, np.random.randint(min_size, max_size))
    return tables
