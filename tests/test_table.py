import pytest
from beachbums.persons import Person
from beachbums.tables import Table


@pytest.fixture
def person():
    return Person({"NAME": "Nompilo"})


@pytest.fixture
def table():
    return Table("Table1", 4)


def test_table_init(table):
    assert table.name == "Table1"
    assert table.capacity == 4
    assert table.people == []


def test_table_seated(table, person):
    table.add_person(person)
    assert table.seated == 1


def test_table_has_space(table, person):
    table.add_person(person)
    assert table.has_space


def test_table_is_full(table, person):
    table.add_person(person)
    table.add_person(person)
    table.add_person(person)
    table.add_person(person)
    assert table.is_full


def test_table_add_person(table, person):
    assert table.add_person(person) == 1
    assert person in table.people


def test_table_add_person_table_full(table, person):
    table.add_person(person)
    table.add_person(person)
    table.add_person(person)
    table.add_person(person)

    with pytest.raises(ValueError):
        table.add_person(person)


def test_table_remove_person(table, person):
    table.add_person(person)
    assert table.remove_person(person) == 0
    assert person not in table.people


def test_table_repr(table, person):
    table.add_person(person)
    assert repr(table) == "Table1(1/4) = ['Nompilo']"


def test_table_str(table, person):
    table.add_person(person)
    assert str(table) == "Table1(1/4) = ['Nompilo']"
