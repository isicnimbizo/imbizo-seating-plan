import datetime
import logging
from functools import partial
from itertools import product
from typing import Dict, List

import numpy as np
import pandas as pd

from .persons import Person
from .tables import Table

logger = logging.getLogger(__name__)
rng = np.random.default_rng()


def process_previous_pairings(
    seating_plans: List[pd.DataFrame], persons: Dict[str, Person]
) -> None:
    """
    Add counts of number of times people have sat together

    Args:
        seating_plans: List of seating plans (loaded from seating-plan files)
        persons: Dictionary of persons (created from names.csv)
    """
    logger.info("Processing previous pairings")
    for seating_plan in seating_plans:
        for table_name, table_frame in seating_plan.groupby("Table"):
            # list of everyone seated at this table in Person object format
            table_persons_list: List[Person] = []
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
    persons: Dict[str, Person],
    tables: Dict[str, Table],
    name_col="NAME",
    save=True,
):
    """
    randomize seating based on TAs being at each table and evenly seat people at each table

    :param people: dataframe containing GROUP (TA, STUDENT, or FACULTY) and people's names
    :param tables: dict of dict with table name: {size, people}
    """
    logger.info("Creating seating plan")
    # filter people based no EXCLUDE column
    excluded_rows = people[(people["EXCLUDE"] == "X") | (people["EXCLUDE"] == "Y")]
    excluded_names = excluded_rows[name_col].values
    excluded_persons = [persons[name] for name in excluded_names]
    people = people[~people["NAME"].isin(excluded_names)]

    tas_df = people[people["GROUP"] == "TA"][name_col]
    random_tas = tas_df.sample(len(tas_df)).tolist()
    ta_persons = [persons[name] for name in random_tas]

    students_df = people[people["GROUP"] == "STUDENT"][name_col]
    random_students = students_df.sample(len(students_df)).tolist()
    student_persons = [persons[name] for name in random_students]

    faculty_df = people[people["GROUP"] == "FACULTY"][name_col]
    random_faculty = faculty_df.sample(len(faculty_df)).tolist()
    faculty_persons = [persons[name] for name in random_faculty]

    # first, sort tables by size (using value of sub-dict key "size") with smallest tables first
    sorted_tables = dict(sorted(tables.items(), key=lambda x: x[1].size))
    # then, combine people with TAs first, then FACULTY, then STUDENTS
    sorted_people = random_tas + random_faculty + random_students

    # ensure everyone is an option
    for ta, other_person in product(ta_persons, student_persons + faculty_persons):
        ta.pairs.setdefault(other_person, 0)

    # keep track of where a person is sitting
    people_dict = {sorted_people[i]: -1 for i in range(len(sorted_people))}

    # create a partial function to add a person to a table, taking in the people dict and table dict
    add_person_to_table = partial(add_person_to_table_generic, people_dict, tables)

    # count size of tables
    total_seats = sum([table.size for table in sorted_tables.values()])
    if total_seats < len(sorted_people):
        raise ValueError(
            f"Not enough seats for all people! {total_seats} < {len(sorted_people)}"
        )
    extra_seats = total_seats - len(sorted_people)
    if len(ta_persons) > len(sorted_tables):
        extra_tas = ta_persons[len(sorted_tables) :]
        ta_persons = ta_persons[: len(sorted_tables)]
    ### SEAT PEOPLE ###
    # first, seat TAs

    # assign tas to table and choose people they haven't sat with yet
    for (table_name, table), ta in zip(sorted_tables.items(), ta_persons):
        # add TA
        added = add_person_to_table(ta, table_name)
        if not added:
            raise ValueError(f"Could not add {ta} to {table_name}")
        # get previous pairs
        options, counts = ta.get_pair_count_for_people_except(
            excluded_persons + faculty_persons + ta_persons
        )
        counts = np.array(counts)
        p = 1 - (counts / counts.sum())
        # assign a probability of 0 to people already at tables
        for i, option in enumerate(options):
            if people_dict[option.name] != -1:
                p[i] = 0
        # make sure p sums to 1
        p = p / p.sum()

        # people to seat.
        # if there are extra seats, don't fill up the tables initially. This would mean the last table could be empty.
        size = (
            table.size
            - table.actual_size
            - (extra_seats + len(faculty_persons) + len(extra_tas))
            // len(sorted_tables)
        )
        # make sure
        if size > np.sum(p > 0):
            size = np.sum(p > 0)
        # choose some people
        choices = rng.choice(
            options,
            size=size,
            replace=False,
            p=p,
        )
        # add them to the table
        for choice in choices:
            added = add_person_to_table(choice, table_name)
            if not added:
                raise ValueError(f"Could not add {choice} to {table_name}")

    # remove assigned people from list of people to seat
    sorted_people = [person for person in sorted_people if people_dict[person] == -1]
    sorted_persons = [persons[person] for person in sorted_people]

    # now, loop through the remaining people and assign them to tables
    table_names = list(sorted_tables.keys())
    table_name = table_names[0]
    for person in sorted_persons:
        # assign starting point for checking in the while loop
        table_start = table_name
        while not add_person_to_table(person, table_name):
            table_name = table_names[
                (table_names.index(table_name) + 1) % len(table_names)
            ]
            if table_name == table_start:
                raise ValueError(f"Could not seat {person}")
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
        raise ValueError(
            f"Some tables have more than one faculty and some tables have no faculty: {multi_faculty_table_count} and {zero_faculty_table_count}"
        )

    # convert to pure string; use sub-dict for easier creation of dataframe
    tables_dict = {k: {"people": [p.name for p in t.people]} for k, t in tables.items()}

    # create a dataframe from the people dict
    seating_plan = pd.DataFrame.from_dict(tables_dict, orient="index").explode("people")
    seating_plan.index.name = "Table"
    seating_plan.reset_index(inplace=True)
    seating_plan.rename(columns={"people": "Name"}, inplace=True)
    if save:
        if not isinstance(save, str):
            file_name = (
                f"seating-plan-{datetime.datetime.now().strftime('%Y-%m-%d')}.csv"
            )
        else:
            file_name = save
        # export to seating-plan-<date>.csv
        seating_plan.to_csv(
            file_name,
            sep=";",
            index=False,
        )

    return seating_plan


def add_person_to_table_generic(
    people_dict, table_dict: Dict[str, Table], person: Person, table_name: str
) -> None:
    """
    Add a person to a table
    """
    if (
        table_dict[table_name].actual_size < table_dict[table_name].size
        and people_dict[person.name] == -1
    ):
        table_dict[table_name].add_person(person)
        # table_dict[table_name]["people"].append(person)
        # table_dict[table_name]["actual_size"] = len(table_dict[table_name]["people"])
        people_dict[person.name] = table_dict[table_name]
        return True
    return False
