from __future__ import annotations

import datetime
import logging
from functools import partial
from itertools import product
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd

from .persons import Person
from .tables import Table

logger = logging.getLogger(__name__)


def process_previous_pairings(
    seating_plans: list[pd.DataFrame], persons: dict[str, Person]
) -> None:
    """
    Add counts of number of times people have sat together

    Args:
        seating_plans: list of seating plans (loaded from seating-plan files)
        persons: dictionary of persons (created from names.csv)
    """
    logger.info("Processing previous pairings")
    for seating_plan in seating_plans:
        groupby_key = "Table" if "Table" in seating_plan else "Group"
        for table_name, table_frame in seating_plan.groupby(groupby_key):
            # list of everyone seated at this table in Person object format
            table_persons_list: list[Person] = []
            for name in table_frame["Name"]:
                if name not in persons:
                    logger.warning(
                        f"{name} has an unknown GROUP, please add them to names.csv"
                    )
                    persons[name] = Person({"NAME": name}, name_col="NAME")
                table_persons_list.append(persons[name])
            # add everyone seated at this table to persons' pairs
            #   note that because pairs are reciprocated (reverse pair is added),
            #   we shorten the list on each iteration
            while len(table_persons_list):
                person = table_persons_list.pop()
                person.add_persons(table_persons_list)


def create_seating_plan(
    people: pd.DataFrame,
    persons: dict[str, Person],
    tables: dict[str, Table],
    name_col: str = "NAME",
    save: bool = True,
    seed: Optional[int] = None,
):
    """
    Create a seating plan based on the given data.

    Args:
        people (pd.DataFrame): The data frame containing information about the people.
        persons (dict[str, Person]): A dictionary of person objects. Objects will be modified.
        tables (dict[str, Table]): A dictionary of table objects. Objects will be modified.
        name_col (str, optional): The column name in the `people` data frame that contains the names. Defaults to "NAME".
        save (bool, optional): Whether to save the seating plan to a file. Defaults to True.
        seed (int, optional): The random seed for reproducibility. Defaults to None.

    Returns:
        pd.DataFrame: The seating plan as a data frame.

    Raises:
        ValueError: If there are not enough seats for all people.
        ValueError: If a person is already seated.
        ValueError: If a person cannot be added to any table.
        ValueError: If a table is full.
        ValueError: If a person could not be seated after trying all tables.
        ValueError: If there are tables with more than one faculty and tables with no faculty.

    Algorithm:
        1. Filter out excluded people based on the "EXCLUDE" column in the `people` data frame.
        2. Randomise Teaching Assistants (TAs) and create a dictionary of TA objects.
        3. Randomise students and create a dictionary of student objects.
        4. Randomise faculty members and create a dictionary of faculty objects.
        5. Randomise guests and create a dictionary of guest objects.
        6. Sort the tables by size in ascending order (smallest first).
        7. Combine all unseated people (TAs, faculty, students, and guests) into one dictionary.
        8. Add options for each TA to sit with other unseated people.
        9. Create a dictionary to keep track of where each person is sitting.
        10. Create a partial function to add a person to a table.
        11. Calculate the total number of seats and check if there are enough seats for all people.
        12. If there are extra TAs, remove them from the dictionary of TAs.
        13. Seat the TAs at the tables.
        14. Remove people with preferences from the dictionaries of faculty, guests, students, and extra TAs.
        15. For each person (preferences first), calculate the weight for each table based on their preferences and the current seating arrangement.
        16. Choose the table with the lowest weight for each person and add them to the table.
        17. If a person cannot be added to any table, raise an error.
        18. Assign the remaining unseated people to tables in a round-robin fashion.
        19. Check if each table has at least one faculty member and no more than one faculty member.
        20. Convert the seating plan to a data frame and save it to a file if specified.

    """
    logger.info("Creating seating plan")
    rng = np.random.default_rng(seed)
    # 1. filter people based on EXCLUDE column
    # excluded_rows = people[(people["EXCLUDE"] == "X") | (people["EXCLUDE"] == "Y")]
    # excluded_names: list[str] = list(excluded_rows[name_col].values)
    excluded_names = {p.name for p in persons.values() if p.exclude}
    people = people[~people["NAME"].isin(excluded_names)]

    # 2 - 5. randomise groups
    def _randomize_group(group: str) -> dict[str, Person]:
        """Randomise a group of people and return a dictionary of Person objects

        Note that this inner function uses nonlocal variables 'people', 'persons', 'name_col' and  'rng'
        """
        group_s = people[people["GROUP"] == group][name_col]
        random_group: list[str] = group_s.sample(frac=1, random_state=rng).tolist()
        group_persons = [persons[name] for name in random_group]
        group_persons = sorted(
            group_persons, key=lambda x: len(x.preferred), reverse=True
        )
        return {p.name: p for p in group_persons}

    ta_persons = _randomize_group("TA")
    student_persons = _randomize_group("STUDENT")
    faculty_persons = _randomize_group("FACULTY")
    guest_persons = _randomize_group("GUEST")

    # 6. sort tables by size (using value of sub-dict key "capacity") with smallest tables first
    sorted_tables = dict(sorted(tables.items(), key=lambda x: x[1].capacity))
    # 7. combine people with TAs first, then FACULTY, then STUDENTS
    unseated_people = ta_persons | faculty_persons | student_persons | guest_persons

    # 8. ensure everyone is an option
    # for ta, other_person in product(
    #     ta_persons.values(),
    #     (student_persons | faculty_persons | guest_persons).values(),
    # ):
    #     ta.add_option(other_person)

    # 9. keep track of where a person is sitting
    people_table_placement: dict[str, Union[int, Table]] = {
        person_name: -1 for person_name in unseated_people.keys()
    }

    # 10. create a partial function to add a person to a table, taking in the people dict and
    # table dict
    add_person_to_table_func = add_person_to_table_factory(
        people_table_placement, tables
    )

    # 11. count size of tables
    total_seats = sum([table.capacity for table in sorted_tables.values()])
    if total_seats < len(unseated_people):
        raise ValueError(
            f"Not enough seats for all people! {total_seats} < {len(unseated_people)}"
        )

    extra_seats = total_seats - len(unseated_people)
    logger.info(f"Total seats: {total_seats}. Extra seats: {extra_seats}")

    # 12. if there are extra TAs, remove ('pop') them from the dictionary of TAs
    # TAs with preferences will be at the beginning of the ta list and extras are selected from the end
    extra_tas = {}
    if len(ta_persons) > (num_sorted_tables := len(sorted_tables)):
        extra_tas = {
            ta_name: ta_persons.pop(ta_name)
            for ta_name in list(ta_persons.keys())[num_sorted_tables:]
        }

    # 13. seat TAs
    for (table_name, table), (ta_name, ta) in zip(
        sorted_tables.items(), ta_persons.items()
    ):
        added = add_person_to_table_func(ta, table_name)
        unseated_people.pop(ta_name)
        if not added:
            raise ValueError(f"Could not add {ta} to {table_name}")

    # 14. put people with preferences in their own 'group' dict for seating next at tables
    persons_w_pref_or_as_pref: dict[str, Person] = {}

    # 14.1 add preferences as mutual
    for p in persons.values():
        if len(p.preferred):
            for person_name in p.preferred:
                persons[person_name].preferred.add(p.name)

    # 14.2 add people with preferences to the dict
    for p in unseated_people.values():
        if len(p.preferred):
            persons_w_pref_or_as_pref[p.name] = p
            for group in [faculty_persons, guest_persons, student_persons, extra_tas]:
                # remove from their usual group to prevent double counting
                if p.name in group:
                    group.pop(p.name)

    # 15. for each person, get the normalised count of people at each table
    # and choose the lowest
    for group in [
        persons_w_pref_or_as_pref,
        faculty_persons,
        guest_persons,
        student_persons,
        extra_tas,
    ]:
        for person_name, person in group.items():
            table_weight: dict[str, float] = {}
            for table_name, table in sorted_tables.items():
                # ignore tables that are full
                if table.is_full:
                    continue
                options, counts = person.get_pair_count_for_people(table.people)
                for option in options:
                    if (
                        person_name in option.preferred
                        or option.name in person.preferred
                        or person.preferred in option.preferred
                    ):
                        logger.debug(
                            f"found a preferred match for {person_name} - {option.name}"
                            f"\n{option.preferred=} - {person.preferred=}"
                        )

                        table_weight[table_name] = -1
                        break
                else:
                    table_weight[table_name] = 0
                if person.is_faculty:
                    for i, option in enumerate(options):
                        if option.is_faculty:
                            if counts[i] == 0:
                                counts[i] = 100
                            else:
                                counts[i] *= 100
                if table_weight[table_name] >= 0:
                    table_weight[table_name] = (table.seated / table.capacity) + (
                        sum(counts) / len(options)
                    )
            # choose the table with the lowest value and that
            min_table_name = min(table_weight, key=table_weight.get)  # type: ignore
            added = add_person_to_table_func(person, min_table_name)
            if logger.getEffectiveLevel() <= logging.DEBUG:
                pretty_tables = "\n".join(
                    [
                        f"{table_weight.get(table_name,-1):.2f}-{table}"
                        for table_name, table in tables.items()
                    ]
                )
                logger.debug(f"added {person} to {min_table_name}.\n{pretty_tables}")
            if not added:
                if tables[min_table_name].is_full:
                    raise ValueError(f"Table '{min_table_name}' is full.")
                elif people_table_placement[person_name] != -1:
                    raise ValueError(f"Person '{person_name}' already seated.")
                raise ValueError(f"Could not add {person} to any table")

    unseated_people = {
        person: persons[person]
        for person in unseated_people
        if people_table_placement[person] == -1
    }
    if unseated_people:
        raise ValueError(f"Could not seat everyone! {unseated_people}")

    # now, loop through the remaining people (including faculty) and
    # assign them to tables
    table_names = list(sorted_tables.keys())
    table_name = table_names[0]
    for person_name, person in unseated_people.items():
        # assign starting point for checking in the while loop
        table_start = table_name
        while not add_person_to_table_func(person, table_name):
            # loop through tables until we find one that works
            table_name = table_names[
                (table_names.index(table_name) + 1) % len(table_names)
            ]
            if table_name == table_start:
                raise ValueError(f"Could not seat {person}, tried all tables")
        table_name = table_names[(table_names.index(table_name) + 1) % len(table_names)]

    # checks
    # 1. check if faculty are distributed to each table
    faculty_table_count = {table_name: 0 for table_name in table_names}
    for table_name, table in sorted_tables.items():
        faculty_table_count[table_name] = len(
            [person for person in table.people if person.group == "FACULTY"]
        )
    # count the tables with more than 1 faculty
    multi_faculty_table_count = [
        table_name for table_name, count in faculty_table_count.items() if count > 1
    ]
    # count the tables with 0 faculty
    zero_faculty_table_count = [
        table_name for table_name, count in faculty_table_count.items() if count == 0
    ]
    if len(multi_faculty_table_count) > 0 and len(zero_faculty_table_count) > 0:
        # log the tables with more than 1 faculty
        err_msg = (
            f"Some tables have more than one faculty and some tables have no faculty."
            f"\nMulti: {multi_faculty_table_count} and Zero:{zero_faculty_table_count}"
        )
        logger.error(err_msg)
        for table_name, table in sorted_tables.items():
            if table_name in multi_faculty_table_count:
                logger.error(f"MULTI -> {table_name}: {table.people}")
            elif table_name in zero_faculty_table_count:
                logger.error(f"ZERO  -> {table_name}: {table.people}")
            else:
                logger.warning(f" OK -> {table_name}: {table.people}")
        raise ValueError(err_msg)

    # convert to pure string; use sub-dict for easier creation of dataframe
    tables_dict = {k: {"people": [p.name for p in t.people]} for k, t in tables.items()}

    # create a dataframe from the table dict
    seating_plan = pd.DataFrame.from_dict(tables_dict, orient="index").explode("people")
    seating_plan.index.name = "Table"
    seating_plan.reset_index(inplace=True)
    seating_plan.rename(columns={"people": "Name"}, inplace=True)
    if save:
        if not isinstance(save, str):
            file_name = (
                f"seating-plan-{datetime.datetime.now().strftime('%Y-%m-%d')}.csv"
            )
            # check if file exists
            if Path(file_name).exists():
                # file name that includes time
                file_name = f"seating-plan-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.csv"
        else:
            file_name = save
        # export to seating-plan-<date>.csv
        seating_plan.to_csv(
            file_name,
            sep=";",
            index=False,
        )

    return seating_plan


def add_person_to_table_factory(
    people_table_placement: dict[str, int | Table], table_options: dict[str, Table]
) -> Callable[[Person, str], bool]:
    """
    Generates a function that adds a person to a table based on the provided `people_table_placement` and `table_options`.

    Args:
        people_table_placement (dict[str, int | Table]): A dictionary that maps table names to either an integer or a Table object. The integer represents the number of available seats at the table.
        table_options (dict[str, Table]): A dictionary that maps table names to Table objects. Each Table object contains additional information about the table.

    Returns:
        Callable[[Person, str], bool]: A callable function that takes a Person object and a table name as parameters, and returns a boolean value indicating whether the person was successfully added to the table.
    """
    return partial(_add_person_to_table_generic, people_table_placement, table_options)


def _add_person_to_table_generic(
    people_table_placement: dict[str, int | Table],
    table_options: dict[str, Table],
    person: Person,
    table_name: str,
) -> bool:
    """
    Add a person to a table.

    Args:
        people_table_placement (dict[str, int | Table]): A dictionary mapping person names to table indices or Table objects.
        table_options (dict[str, Table]): A dictionary mapping table names to Table objects.
        person (Person): The person to be added to the table.
        table_name (str): The name of the table to which the person should be added.

    Returns:
        bool: True if the person was successfully added to the table, False otherwise.
    """
    if (
        table_options[table_name].has_space
        and people_table_placement[person.name] == -1
    ):
        table_options[table_name].add_person(person)
        people_table_placement[person.name] = table_options[table_name]
        return True
    return False
