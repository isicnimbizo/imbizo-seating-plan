# FILEPATH: /Users/ccurrin/dev/imbizo-seating-plan/tests/test_group_plan.py

import pandas as pd
import pytest
from beachbums.persons import Person
from beachbums.group_plan import create_groups_based_on_background


@pytest.fixture
def persons():
    return {
        "Nompilo": Person(
            {
                "NAME": "Nompilo",
                "GROUP": "STUDENT",
                "Math": 10,
                "Neuro": 3,
                "Physics": 2,
            }
        ),
        "Anna": Person(
            {"NAME": "Anna", "GROUP": "STUDENT", "Math": 8, "Neuro": 10, "Physics": 3}
        ),
        "Yugi": Person(
            {"NAME": "Yugi", "GROUP": "STUDENT", "Math": 3, "Neuro": 4, "Physics": 4}
        ),
        "Ash": Person(
            {"NAME": "Ash", "GROUP": "STUDENT", "Math": 1, "Neuro": 1, "Physics": 1}
        ),
    }


def test_with_match(persons: dict[str, Person]):
    # sort persons by "Math"
    sorted_persons_list = sorted(
        persons.values(),
        key=lambda x: x.backgrounds["Math"] if "Math" in x.backgrounds else 0,
    )
    sorted_persons = {p.name: p for p in sorted_persons_list}

    # Test with matching backgrounds
    groups: pd.DataFrame = create_groups_based_on_background(
        persons, "Math", 2, match=True, seed=42
    )
    assert groups["Group"].nunique() == 2
    # when matching, top and bottom in Math should be in the same group
    groups_ = (
        groups.groupby("Group")
        .apply(lambda x: x["Name"].tolist())
        .reset_index(name="Names")
    )
    # best and worst first
    assert groups_.iloc[0]["Names"] == ["Nompilo", "Ash"]
    assert groups_.iloc[1]["Names"] == ["Anna", "Yugi"]


def test_with_non_match(persons: dict[str, Person]):
    # Test with non-matching backgrounds
    groups = create_groups_based_on_background(persons, "Math", 2, match=False, seed=42)
    assert groups["Group"].nunique() == 2
    # when not matching, top and bottom in Math should be in different groups
    groups_ = (
        groups.groupby("Group")
        .apply(lambda x: x["Name"].tolist())
        .reset_index(name="Names")
    )
    assert groups_.iloc[0]["Names"] == ["Nompilo", "Anna"]
    assert groups_.iloc[1]["Names"] == ["Yugi", "Ash"]


def test_with_large_group(persons: dict[str, Person]):
    # Test with group size larger than number of persons
    with pytest.raises(ValueError):
        create_groups_based_on_background(persons, "Math", 10, seed=42)


def test_with_fake_background(persons: dict[str, Person]):
    # Test with non-existing background
    with pytest.raises(ValueError):
        create_groups_based_on_background(persons, "NonExisting", 2, seed=42)
