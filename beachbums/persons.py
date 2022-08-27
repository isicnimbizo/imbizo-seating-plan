import copy
import logging
from typing import Dict, List, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

def create_person_objects(
    people: pd.DataFrame,
    name_col="NAME",
) -> Dict[str, "Person"]:
    """
    Create person objects from people dataframe
    """
    logger.info("Creating person objects")
    return {
        person_record_dict[name_col]: Person(person_record_dict, name_col=name_col)
        for person_record_dict in people.to_dict("records")
    }


class Person:
    """
    Class for a Person that represents a person in the beachbums database.

        Attributes:
            name (str): the person's name
            group (str): the person's group (TA, STUDENT, or FACULTY)
            background (str): the person's background (optional)
            exclude (bool): whether the person is excluded from the beachbums
            pairs (dict): a dict of dicts where the key is the person's name and the value is a dict of the number of times they've sat with someone

        Methods:
            add_pair(other_person, reciprocate=True): add a count of 1 to the pair count for this person and the other person
            add_persons(other_persons): add a count of 1 to the pair count for this person and the other persons
            get_pairs(): return the list of people with whom this person has sat
            get_pair_count(): return the total number of times this person has sat with someone
            get_pair_count_for_person(other_person): return the number of times this person has sat with the other person
            get_pair_count_for_people(other_people): return the number of times this person has sat with all of the other people
            get_pair_count_for_people_except(other_people): return the number of times this person has sat with all of the other people except the other people
            get_pair_count_for_people_except_person(other_person): return the number of times this person has sat with all of the other people except the other person
            get_pair_count_for_people_except_people(other_people): return the number of times this person has sat with all of the other people except the other people
    """

    def __init__(
        self,
        record: dict,
        name_col="NAME",
        group_col="GROUP",
        background_col="BACKGROUND",
    ):
        """
        Initialize a Person object from a record dict.

        Expected record dict format:
        {
            "NAME": "John",
            "GROUP": "STUDENT",
            "BACKGROUND": "Maths",
        }

        Args:
            record (dict): the record dict from the people dataframe
            name_col (str): the name column name
            group_col (str): the group column name
            background_col (str): the background column name

        """
        record = copy.deepcopy(record)
        self.name = record.pop(name_col)
        self.group = record.pop(group_col, None)
        self.background = record.pop(background_col, None)
        self.exclude = record.pop("EXCLUDE", False)
        self._unique_name = f"{self.name}_{hash(self.name)}"
        # leave the rest of the record dict as is
        self.props = record

        # initialize the pair count dict
        self.pairs: Dict[Person, int] = {}

    def __repr__(self):
        return f"{self.name}"

    def __str__(self):
        return f"{self.name}"

    def __eq__(self, other):
        return self._unique_name == other._unique_name

    def __hash__(self):
        return hash(self.name)

    def add_persons(self, other_persons: List["Person"]):
        """Add a count of 1 to the pair count for this person and the other persons

        Args:
            other_persons (List[Person]): the other persons with which to pair

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

    def get_pairs(self):
        return self.pairs.keys()

    def get_total_pair_count(self):
        return sum(self.pairs.values())

    def get_pair_count_for_person(self, other_person):
        return self.pairs[other_person] if other_person in self.pairs else 0

    def get_pair_count_for_people(self, other_people):
        pairs = {
            other_person: self.get_pair_count_for_person(other_person)
            for other_person in other_people
        }
        return zip(*pairs.items())

    def get_pair_count_for_people_except(
        self, other_people: List["Person"]
    ) -> Tuple[List["Person"], List[int]]:
        if isinstance(other_people, Person):
            other_people = [other_people]
        pairs = {
            other_person: self.get_pair_count_for_person(other_person)
            for other_person in self.get_pairs()
            if other_person not in other_people
        }
        return zip(*pairs.items())
