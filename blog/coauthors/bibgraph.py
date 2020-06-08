
import pickle
import re
import pandas as pd
import networkx as nx
import bibtexparser
from nameparser import HumanName

DUMP = "expe/biblio.bib"
AUTHOR_SEPARATOR = " and | and\n|\nand "


def parse_name(in_name):
    name = in_name.strip()
    name = name.replace("\n", " ")
    return name


def count_authors(db):
    counts = dict()
    for entrie in db.entries:
        names = entrie["author"]
        for author in parse_authors(names):
            name = parse_name(author)
            counts[name] = 1 + counts.get(name, 0)
    return counts


def key_name(name):
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
    id_node = 0
    authors = load_authors(database)
    correspondance = dict()
    for auth in authors:
        id_node += 1
        gx.add_node(id_node)
        correspondance[auth] = id_node
    return correspondance


def _add_edges(gx, database, correspondance):
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
    gx = nx.Graph()
    id_node = 0
    correspondance = _add_nodes(gx, database)
    _add_edges(gx, database, correspondance)
    return gx, correspondance


def parse_authors(names):
    return re.split(AUTHOR_SEPARATOR, names)


def load_bibtex(bibname):
    parser = bibtexparser.bparser.BibTexParser(common_strings=True)
    with open(bibname) as bibtex_file:
        bib_database = bibtexparser.load(bibtex_file, parser)
    return bib_database


def load_years(db):
    counts = dict()
    for entrie in db.entries:
        year = entrie["year"]
        counts[year] = 1 + counts.get(year, 0)
    return counts


def dump_authors(db):
    d = load_authors(db)
    authors = d.keys()
    with open("authors2.txt", "w") as f:
        for author in authors:
            f.write("%s\n" % author)


def dump_database(db):
    with open("bibtex.pickle", "wb") as f:
        pickle.dump(db, f)


def dump_metadata(key_authors, key_id):
    with open("metadata.txt", "w") as f:
        for k in key_authors:
            id_ = key_id[k]
            f.write("%s;%s;%s\n" % (k, id_, key_authors[k]))


if __name__ == "__main__":
    # db = load_bibtex(DUMP)
    with open("bibtex.pickle", "rb") as f:
        db = pickle.load(f)
    print("Num entries: ", len(db.entries))
    d = load_authors(db)
    g, c = build_graph(db)

