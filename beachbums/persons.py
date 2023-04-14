from __future__ import annotations

import copy
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

default_background_cols = {
    "Math",
    "Physics",
    "Stats",
    "Chem",
    "Bio",
    "Programming",
    "Neuro",
}


def create_person_objects(
    people: pd.DataFrame,
    name_col="NAME",
    background_cols=default_background_cols,
) -> dict[str, "Person"]:
    """
    Create person objects from people dataframe
    """
    logger.info("Creating person objects")
    return {
        person_record_dict[name_col]: Person(
            person_record_dict, name_col=name_col, background_cols=background_cols
        )
        for person_record_dict in people.to_dict("records")
    }


def create_adjacency_matrix(persons: dict[str, "Person"]):
    """
    Create adjacency matrix for persons
    """
    logger.info("Creating adjacency matrix")
    # create pandas dataframe with person names as index and columns as person names
    person_names = [person.name for person in persons.values()]
    adjacency_matrix = pd.DataFrame(
        index=person_names,
        columns=person_names,
        data=np.zeros((len(persons), len(persons))),
    )
    for p_name, person in persons.items():
        for other_person, count in person.pairs.items():
            adjacency_matrix.loc[person.name, other_person.name] = count
    return adjacency_matrix


class Person:
    """
    Class for a Person that represents a person in the beachbums database.

        Attributes:
            name (str): the person's name
            group (str): the person's group (TA, STUDENT, or FACULTY)
            backgrounds (dict): the person's backgrounds (optional)
            exclude (bool): whether the person is excluded from the beachbums
            pairs (dict): a dict of dicts where the key is the person's name and the value is a dict of the number of times they've sat with someone

        Methods:
            add_pair(other_person, reciprocate=True): add a count of 1 to the pair count for this person and the other person
            add_persons(other_persons): add a count of 1 to the pair count for this person and the other persons
            get_pairs(): return the list of people with whom this person has sat
            get_total_pair_count(): return the total number of times this person has sat with someone
            get_pair_count_for_person(other_person): return the number of times this person has sat with the other person
            get_pair_count_for_people(other_people): return the number of times this person has sat with all of the other people
            get_pair_count_for_people_except(other_people): return the number of times this person has sat with all of the other people except the other people
    """

    def __init__(
        self,
        record: dict,
        name_col="NAME",
        group_col="GROUP",
        background_cols=default_background_cols,
    ):
        """
        Initialize a Person object from a record dict.

        Expected record dict format:
        {
            "NAME": "John",
            "GROUP": "STUDENT",
            "Maths": 10,
            "Physics": 9,
            "Neuroscience": 5,
            "EXCLUDE": False
        }

        Args:
            record (dict): the record dict from the people dataframe
            name_col (str): the name column name
            group_col (str): the group column name
            background_cols (Optional[set[str]]): the background column names

        """
        record = copy.deepcopy(record)
        self.name = record.pop(name_col)
        self.group = record.pop(group_col, None)
        # add background values if they exist (non-nan when converted to float)
        self.backgrounds = {
            col: value
            for col in background_cols
            if col in record and not np.isnan(value := float(record.pop(col)))
        }
        self.preferred = record.pop("PREFERRED", None)
        self.exclude = record.pop("EXCLUDE", False)
        self._unique_name = f"{self.name}_{hash(self.name)}"
        # leave the rest of the record dict as is
        self.props = record

        # initialize the pair count dict
        self.pairs: dict[Person, int] = {}

    def __repr__(self) -> str:
        return self.name

    def __str__(self):
        return self.name

    def __eq__(self, other):
        return self._unique_name == other._unique_name

    def __hash__(self) -> int:
        return (
            hash(self.name)
            + hash(self.group)
            + hash(str(self.backgrounds))
            + hash(self.exclude)
            + hash(str(self.props))
        )

    @property
    def is_faculty(self) -> bool:
        return self.group == "FACULTY"

    @property
    def is_ta(self) -> bool:
        return self.group == "TA"

    @property
    def is_student(self) -> bool:
        return self.group == "STUDENT"

    def add_persons(self, other_persons: list["Person"]):
        """Add a count of 1 to the pair count for this person and the other persons

        Args:
            other_persons (list[Person]): the other persons with which to pair

        """
        for other_person in other_persons:
            self.add_pair(other_person, reciprocate=True)

    def add_pair(self, other_person: "Person", reciprocate=True):
        """Add a count of 1 to the pair count for this person and the other person

        Args:
            other_person (Person): the other person with which to pair
            reciprocate (bool): whether to add a pair from the other person's side as well
                In other ways, add a count for the current person to the other person's pair count

        """
        if self == other_person:
            return
        if other_person not in self.pairs:
            self.pairs[other_person] = 1
        else:
            self.pairs[other_person] += 1

        if reciprocate:
            # prevent infinite recursion
            other_person.add_pair(self, reciprocate=False)

    def add_option(self, other_person: "Person"):
        """Add the other person as an option for pairs (default pair count of 0)"""
        if self == other_person:
            return
        self.pairs.setdefault(other_person, 0)

    @property
    def pretty_pairs(self):
        """return pairs as strings instead of Person objects"""
        return {other_person.name: count for other_person, count in self.pairs.items()}

    def get_pairs(self):
        return self.pairs.keys()

    def get_total_pair_count(self):
        return sum(self.pairs.values())

    def get_pair_count_for_person(self, other_person: "Person"):
        return self.pairs[other_person] if other_person in self.pairs else 0

    def get_pair_count_for_people(self, other_people: set["Person"] | list["Person"]):
        pairs = {
            other_person: self.get_pair_count_for_person(other_person)
            for other_person in other_people
        }
        return list(pairs.keys()), list(pairs.values())

    def get_pair_count_for_everyone_except(
        self, except_other_people: set["Person"] | list["Person"] = set()
    ) -> tuple[list["Person"], list[int]]:
        if isinstance(except_other_people, Person):
            except_other_people = set(except_other_people)
        pairs = {
            other_person: self.get_pair_count_for_person(other_person)
            for other_person in self.get_pairs()
            if other_person not in except_other_people
        }
        return list(pairs.keys()), list(pairs.values())
