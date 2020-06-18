#!/usr/bin/python

import re
import pickle
import networkx as nx
import bibtexparser
from nameparser import HumanName

DUMP = "hal_biblio.bib"
OUTPUT_NAME = "coauthors.graphml"
AUTHOR_SEPARATOR = " and | and\n|\nand "

# Implement some utils functions
def load_bibtex(bibname):
    "Load bibtex database from bibtex file `bibname`."
    parser = bibtexparser.bparser.BibTexParser(common_strings=True)
    with open(bibname) as bibtex_file:
        bib_database = bibtexparser.load(bibtex_file, parser)
    return bib_database


def parse_authors(names):
    "Parse a list of authors separated by a `and` substring."
    return re.split(AUTHOR_SEPARATOR, names)


def parse_name(in_name):
    "Preprocess name"
    name = in_name.strip()
    return name


def count_authors(db):
    "Count the authors appearing in the bibtex database."
    counts = dict()
    for entrie in db.entries:
        names = entrie["author"]
        for author in parse_authors(names):
            name = parse_name(author)
            counts[name] = 1 + counts.get(name, 0)
    return counts


def hist_years(db):
    "Return an histogram to show when were published the articles in `db`."
    counts = dict()
    for entrie in db.entries:
        year = entrie["year"]
        counts[year] = 1 + counts.get(year, 0)
    years = list(counts.keys())
    years.sort()
    for year in years:
        print("%s: %s" % (year, counts[year]))
    return counts


# Dump functions

def dump_authors(db):
    d = load_authors(db)
    authors = d.keys()
    with open("authors.txt", "w") as f:
        for author in authors:
            f.write("%s\n" % author)


def dump_database(db):
    with open("bibtex.pickle", "wb") as f:
        pickle.dump(db, f)


def dump_metadata(key_authors, key_id):
    with open("metadata.txt", "w") as f:
        for k in key_authors:
            id_ = key_id[k]
            f.write("%s;%s;%s\n" % (id_, k, key_authors[k]))


# Implement core functions

def key_name(name):
    "Return a key associated to the `name` passed in input."
    # parse name
    parsed = HumanName(name)
    if len(parsed.first) > 0:
        first_name = parsed.first[0]
        # Key is lowercased
        key = f"{parsed.last.lower()} {first_name.lower()}"
        return key
    else:
        return name.lower()


def load_authors(db):
    """Return a dictionnary (key -> matching authors).
    The hash is given by the function `key_name`.

    """
    authors = dict()
    for entrie in db.entries:
        names = entrie["author"]
        for author in parse_authors(names):
            name = parse_name(author)
            key = key_name(name)
            val = authors.get(key)
            if isinstance(val, list) and name not in val:
                val.append(name)
            else:
                authors[key] = [name]
    return authors


def _add_nodes(gx, database):
    "For each author in `database`, add a new node to the graph `gx`."
    id_node = 0
    authors = load_authors(database)
    correspondance = dict()
    for auth in authors:
        id_node += 1
        gx.add_node(id_node)
        correspondance[auth] = id_node
    return correspondance


def _add_edges(gx, database, correspondance):
    """
    For each article in `database`, add to the graph `gx` new edges
    corresponding to the coauthorship between authors.
    """
    for entrie in database.entries:
        names = entrie["author"]
        authors = []
        # Parse names
        for author in parse_authors(names):
            name = parse_name(author)
            authors.append(name)
        # Add all edges
        for name in authors:
            k1 = key_name(name)
            for coname in authors:
                k2 = key_name(coname)
                if k1 != k2:
                    o = correspondance[k1]
                    d = correspondance[k2]
                    gx.add_edge(o, d)


def build_graph(database):
    "Build a networkx graph from bibtex database."
    gx = nx.Graph()
    correspondance = _add_nodes(gx, database)
    _add_edges(gx, database, correspondance)
    return gx, correspondance


if __name__ == "__main__":
    db = load_bibtex(DUMP)
    # In case your database was pickled before:
    # with open("bibtex.pickle", "rb") as f:
    #     db = pickle.load(f)
    print("Num entries: ", len(db.entries))
    # Load authors
    d = load_authors(db)
    # Build coauthorship network
    g, c = build_graph(db)
    # Save graph
    nx.write_graphml(g, "coauthors.graphml")
    # Save metadata
    dump_metadata(d, c)

