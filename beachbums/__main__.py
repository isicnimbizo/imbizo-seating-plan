"""
create seating plan using names from names.csv and output to seating-plan.csv
"""
from __future__ import annotations

import argparse
import logging
from multiprocessing import Pool
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from beachbums.data import load_past_groupings, load_people, load_table_layout
from beachbums.group_plan import create_groups_based_on_background
from beachbums.persons import (
    create_adjacency_matrix,
    create_person_objects,
    default_background_cols,
)
from beachbums.seating_plan import create_seating_plan, process_previous_pairings
from beachbums.tables import create_random_tables, define_table_layout

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="create seating plan")
    parser.add_argument("-n", "--names", help="path to names.csv", default="names.csv")
    parser.add_argument(
        "-p",
        "--past-groups",
        help="path to past-pairs-*.csv or seating-plan-*.csv files",
        default="seating-plan-*.csv",
        nargs="*",
    )
    # add report flag
    parser.add_argument(
        "-r", "--report", help="generate a report? (flag)", action="store_true"
    )
    # output file
    parser.add_argument(
        "-o", "--output", help="path to output seating plan file", default=True
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
        "--random", help="create random seating plan", action="store_true"
    )
    table_options.add_argument(
        "-c",
        "--num-tables",
        help="number of tables (for random generation)",
        type=int,
        default=6,
    )
    table_options.add_argument(
        "--min-size",
        help="minimum number of people to seat (for random generation)",
        type=int,
        default=40,
    )
    table_options.add_argument(
        "--var-size",
        help="minimum variation in size of tables (for random generation)",
        type=int,
        default=2,
    )
    # add group for group students based on background
    background_options = parser.add_argument_group(
        title="grouping of students based on background instead of at a table"
    )
    background_options.add_argument(
        "--background",
        help="which background in names to group on, e.g. Math",
        default=None,
        choices=sorted(default_background_cols),
    )
    # add argument for sorting of backgrounds or reverse mixing
    background_options.add_argument(
        "--match",
        help=(
            "match people based on inverse background skill. (default: True) "
            "People with high skill on the background are likely to be "
            "paired with low skill. "
            "False is similar skill."
        ),
        choices=["True", "False"],
        default=True,
        type=bool,
    )

    background_options.add_argument(
        "-g",
        "--group-size",
        help="number of students in each group",
        type=int,
        default=2,
    )

    # add verbosity argument group
    verbosity_options = parser.add_argument_group(title="debug options")
    verbosity_options.add_argument(
        "-v", "--verbose", help="increase output verbosity", action="store_true"
    )
    verbosity_options.add_argument(
        "-vv", "--very-verbose", help="increase output verbosity", action="store_true"
    )
    verbosity_options.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "set the seed used when ordering people."
            " Default of None keeps things random"
        ),
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
    for ignore_mods in ["matplotlib", "seaborn"]:
        logging.getLogger(ignore_mods).setLevel(logging.WARNING)
    logger.info(f"log level: {logging.getLevelName(log_level)}")

    if args.random:
        logger.info("creating random seating plan")
        tables = create_random_tables(
            num_tables=args.num_tables,
            min_total_size=args.min_size,
            min_max_variation=args.var_size,
        )

    args = parser.parse_args()

    if args.random:
        tables = create_random_tables(
            num_tables=args.num_tables,
            min_total_size=args.min_size,
            min_max_variation=args.var_size,
        )
    else:
        try:
            layout = load_table_layout(args.table_layout)
            tables = define_table_layout(layout)
        except FileNotFoundError:
            logger.error(f"file not found: {args.table_layout}")
            print("-" * 80)
            parser.print_help()
            print("\n" + "-" * 80)
            logger.error(f"file not found: {args.table_layout}")
            sys.exit(1)

    past_groups_search_str = args.past_groups
    if args.background and "seating" in past_groups_search_str:
        past_groups_search_str = "group-plan-*.csv"

    past_groupings_files = Path(".").glob(past_groups_search_str)
    if not past_groupings_files:
        logger.error(f"file not found: {past_groups_search_str}")
        print("-" * 80)
        parser.print_help()
        print("\n" + "-" * 80)
        logger.error(f"file not found: {past_groups_search_str}")
        sys.exit(1)

    logger.info("loading previous groups")
    with Pool() as pool:
        past_groupings = pool.map(load_past_groupings, past_groupings_files)
    # past_groupings = [load_past_groupings(fname) for fname in past_groupings_files]

    try:
        people = load_people(args.names)
    except FileNotFoundError:
        logger.error(f"file not found: {args.names}")
        print("-" * 80)
        parser.print_help()
        print("\n" + "-" * 80)
        logger.error(f"file not found: {args.names}")
        sys.exit(1)

    persons = create_person_objects(people)

    # process previous groupings to update person objects
    # each person object has a .pairs dict that stores the number of times
    # they have been paired with each other person
    process_previous_pairings(past_groupings, persons)

    if args.report:
        # create adjacency matrix of past pairs using person objects' .pairs dict
        adjacency_matrix = create_adjacency_matrix(persons)
        adjacency_matrix.fillna(0, inplace=True)
        adjacency_matrix.to_csv("report.csv")
        with sns.plotting_context("paper", font_scale=0.5):  # type: ignore
            fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
            # plot the 0 values as off-white
            # sns.heatmap(
            #     adjacency_matrix,
            #     ax=ax,
            #     square=True,
            #     mask=adjacency_matrix != 0,
            #     linewidths=0,
            #     cmap="Greys",
            #     # annot=True,
            #     cbar=False,
            # )

            # create a logaritmic scale for the heatmap
            cmap = sns.cubehelix_palette(
                start=0.5,
                rot=-0.75,
                as_cmap=True,
            )

            sns.heatmap(
                adjacency_matrix,
                ax=ax,
                square=True,
                mask=adjacency_matrix == 0,
                linewidths=0.1,
                cmap=cmap,
                # annot=True,
            )
            # grey with an alpha
            grey_alpha = (0.5, 0.5, 0.5, 0.1)
            ax.set_facecolor(grey_alpha)
            fig.savefig("report.png")
    else:
        save = args.output
        logger.info(f"{save=}")
        if args.background:
            seating_plan = create_groups_based_on_background(
                persons,
                args.background,
                args.group_size,
                match=args.match,
                save=save,
            )
            group_column = "Group"
            group_header = "Group {}: "
            join_str = " and "
            next_line = "\n"
        else:
            seating_plan = create_seating_plan(
                people, persons, tables, save=args.output or args.output is not None
            )
            group_column = "Table"
            group_header = "# Table {}\n"
            join_str = "\n"
            next_line = "\n\n" + "-" * 80 + "\n\n"

        seating_plan_text = ""

        parse_skip_n_chars = len(group_column)
        seating_plan["number"] = seating_plan[group_column].apply(
            lambda x: int(x[parse_skip_n_chars:])
        )

        for group_value in sorted(seating_plan["number"].unique()):
            seating_plan_text += group_header.format(group_value)
            names = seating_plan[seating_plan["number"] == group_value].Name.values
            names_str = join_str.join(names)
            seating_plan_text += names_str + next_line

        if args.output:
            with open("pairing text.txt", "w") as f:
                f.write(seating_plan_text)

        print(seating_plan_text)
