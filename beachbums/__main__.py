"""
create seating plan using names from names.csv and output to seating-plan.csv
"""
import argparse
import logging
from pathlib import Path
import sys
from typing import Dict

import numpy as np
import pandas as pd

from beachbums.data import load_people, load_past_groupings, load_table_layout
from beachbums.persons import Person, create_person_objects
from beachbums.seating_plan import create_seating_plan, process_previous_pairings
from beachbums.tables import create_random_tables, define_table_layout

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="create seating plan")
    parser.add_argument("-n", "--names", help="path to names.csv", default="names.csv")
    parser.add_argument(
        "-p",
        "--past-pairs",
        help="path to past-pairs.csv or past-pairs-*.csv files",
        default="seating-plan-*.csv",
        nargs="*",
    )
    # output file
    parser.add_argument(
        "-o", "--output", help="path to output seating plan file", default=None
    )
    # add group for table options
    table_options = parser.add_argument_group(title="table options")
    table_options.add_argument(
        "-t",
        "--table-layout",
        help="path to table-layout.csv",
        default="table-layout.csv",
    )
    table_options.add_argument(
        "-r", "--random", help="create random seating plan", action="store_true"
    )
    table_options.add_argument(
        "-c", "--num-tables", help="number of tables", type=int, default=6
    )
    table_options.add_argument(
        "-ms",
        "--min-size",
        help="minimum number of people to seat",
        type=int,
        default=40,
    )
    table_options.add_argument(
        "-var",
        "--variation",
        help="minimum variation in size of tables",
        type=int,
        default=2,
    )
    # add group for group students based on background
    background_options = parser.add_argument_group(
        title="grouping of students based on background instead of at a table"
    )
    background_options.add_argument(
        "-b", "--background", help="which background in names to group on", default=None
    )
    # add argument for sorting of backgrounds or reverse mixing
    background_options.add_argument(
        "-m",
        "--match",
        help=(
            "match people based on inverse background skill. "
            "People with high skill on the background are likely to be paired with low skill. "
            "False is similar skill."
        ),
        default=True,
    )

    background_options.add_argument(
        "-g",
        "--group-size",
        help="number of students in each group",
        type=int,
        default=2,
    )

    # add verbosity argument group
    verbosity_options = parser.add_argument_group(title="table options")
    verbosity_options.add_argument(
        "-v", "--verbose", help="increase output verbosity", action="store_true"
    )
    verbosity_options.add_argument(
        "-vv", "--very-verbose", help="increase output verbosity", action="store_true"
    )
    args = parser.parse_args()

    log_level = logging.WARNING  # default
    if args.verbose:
        log_level = logging.INFO
    if args.very_verbose:
        log_level = logging.DEBUG
    # set colorama logging format
    import colorama
    colorama.init()
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logger.info(f"log level: {log_level}")

    if args.random:
        logger.info("creating random seating plan")
        tables = create_random_tables(
            num_tables=args.num_tables,
            min_total_size=args.min_size,
            min_max_variation=args.min_variation,
        )

    args = parser.parse_args()

    if args.random:
        tables = create_random_tables(
            num_tables=args.num_tables,
            min_total_size=args.min_size,
            min_max_variation=args.min_variation,
        )
    else:
        try:
            layout = load_table_layout(args.table_layout)
            tables = define_table_layout(layout)
        except FileNotFoundError as e:
            logger.error(f"file not found: {args.table_layout}")
            print("-" * 80)
            parser.print_help()
            print("\n" + "-" * 80)
            logger.error(f"file not found: {args.table_layout}")
            sys.exit(1)

    seating_plan_files = Path(".").glob("seating-plan-*.csv")
    if not seating_plan_files:
        logger.error(f"file not found: seating-plan-<DATE>.csv")
        print("-" * 80)
        parser.print_help()
        print("\n" + "-" * 80)
        logger.error(f"file not found: seating-plan-<DATE>.csv")
        sys.exit(1)

    logger.info("loading previous seating plan or pairs")
    past_groupings = [load_past_groupings(fname) for fname in seating_plan_files]

    try:
        people = load_people(args.names)
    except FileNotFoundError as e:
        logger.error(f"file not found: {args.names}")
        print("-" * 80)
        parser.print_help()
        print("\n" + "-" * 80)
        logger.error(f"file not found: {args.names}")
        sys.exit(1)

    persons = create_person_objects(people)

    process_previous_pairings(past_groupings, persons)
    save = args.output or args.output is None
    logger.info(f"{save=}")
    seating_plan = create_seating_plan(people, persons, tables, save=args.output or args.output is not None)
    print(seating_plan)
