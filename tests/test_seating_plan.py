"""File to test the seating plan methods using PyTest
"""
import logging
import numpy as np

import pandas as pd
import pytest

from beachbums.__main__ import load_past_groupings, load_people, load_table_layout
from beachbums.persons import Person, create_person_objects
from beachbums.seating_plan import create_seating_plan
from beachbums.tables import Table, create_random_tables

logger = logging.getLogger(__name__)


@pytest.fixture
def people() -> pd.DataFrame:
    """
    Load people from names.csv
    """
    return load_people(file_name="tests/data/test_names.csv")


@pytest.fixture
def prev_seating_plan() -> pd.DataFrame:
    """
    Load seating plan from seating-plan.csv
    """
    return load_past_groupings(file_name="tests/data/test-seating-plan.csv")


@pytest.fixture
def tables() -> dict[str, Table]:
    """
    Load table layout from table-layout.csv
    """
    return create_random_tables(6, min_total_size=40, min_max_variation=0)


@pytest.mark.parametrize("seed", [29, 770, None, 128, 1234])
def test_create_seating_plan(
    seed,
    people: pd.DataFrame,
    tables: dict[str, Table],
    prev_seating_plan: pd.DataFrame,
):
    """
    Test create_seating_plan
    """
    # create list of Person objects
    persons = create_person_objects(people, background_cols={})

    # create seating plan
    seating_plan: pd.DataFrame = create_seating_plan(
        people,
        persons,
        tables,
        save=False,
        seed=seed,
    )

    # check that all people are seated (excluded people are not in the seating plan)
    assert seating_plan["Name"].nunique() == len(
        {p.name for p in persons.values() if not p.exclude}
    )

    # check that all tables are used
    assert len(seating_plan["Table"].unique()) == len(tables)

    # check that no table is over capacity
    for table_name, table in tables.items():
        table_size = len(seating_plan[seating_plan["Table"] == table_name])
        assert table_size <= table.capacity

    # check that each table has a TA
    ta_persons = [person for person in persons.values() if person.group == "TA"]
    for table_name, table in tables.items():
        assert any(
            [ta in table.people for ta in ta_persons]
        ), f"Table {table_name} has no TA"

    # check excluded people are not in the seating plan
    assert not any(
        [
            person.name in seating_plan["Name"].tolist()
            for person in persons.values()
            if person.exclude
        ]
    )

    # check preferred seating
    # In the test_names Melissa -> Chris -> Emma & Farai are the preferred connections
    # find Chris' table and check Melissa, Emma, and Farai are in that table
    chris_table_name = seating_plan[seating_plan["Name"] == "Chris"]["Table"].tolist()[
        0
    ]
    chris_table = tables[chris_table_name]
    assert all(
        [
            person in chris_table.people
            for person in [persons["Melissa"], persons["Emma"], persons["Farai"]]
        ]
    ), f"Tables are not preferred seating. {seed=}. {persons['Chris'].preferred}. \nSeating plan: {seating_plan}"


def test_seating_plan_not_enough_space(people: pd.DataFrame, tables: dict[str, Table]):
    """
    Test create_seating_plan
    """
    # create list of Person objects
    persons = create_person_objects(people, background_cols={})

    # remove a table
    tables.popitem()

    # create seating plan
    with pytest.raises(ValueError):
        create_seating_plan(people, persons, tables, save=False)


# repeat test 5 times to ensure randomness is working


def test_seating_plan_more_than_enough_tas(
    people: pd.DataFrame, tables: dict[str, Table]
):
    """Test create_seating_plan"""
    # add one more TA
    people = pd.concat(
        [people, pd.DataFrame({"NAME": "Extra TA", "GROUP": "TA"}, index=[0])],
        ignore_index=True,
    )

    persons = create_person_objects(people, background_cols={})

    # create seating plan
    seating_plan = create_seating_plan(people, persons, tables, save=False)

    # check that all people are seated (excluded people are not in the seating plan)
    assert seating_plan["Name"].nunique() == len(
        {p.name for p in persons.values() if not p.exclude}
    )

    # check that all tables are used
    assert len(seating_plan["Table"].unique()) == len(tables)

    # check that no table is over capacity
    for table_name, table in tables.items():
        table_size = len(seating_plan[seating_plan["Table"] == table_name])
        assert table_size <= table.capacity

    # check that each table has a TA
    ta_persons = [person for person in persons.values() if person.group == "TA"]
    for table_name, table in tables.items():
        assert any(
            [ta in table.people for ta in ta_persons]
        ), f"Table {table_name} has no TA"

    # check excluded people are not in the seating plan
    assert not any(
        [
            person.name in seating_plan["Name"].tolist()
            for person in persons.values()
            if person.exclude
        ]
    )
