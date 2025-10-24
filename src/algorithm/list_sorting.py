
# create four items with imaginary lenghts of edges
def create_list():
    puzzleItem1 = [10, 12, 14, 10]
    puzzleItem2 = [9, 10, 10, 16]
    puzzleItem3 = [12, 10, 10, 16]
    puzzleItem4 = [10, 10, 14, 9]

    return [puzzleItem1, puzzleItem2, puzzleItem3, puzzleItem4]


def sort_list():
    lists = create_list()
    for lst in lists:
        lst.sort()
    return lists


def print_lists():
    lists = sort_list()
    for idx, liste in enumerate(lists, start=1):
        print(f"Liste {idx}:", liste)


def find_shared_values_with_locations():
    lists = create_list()
    value_locations = {}

    for i, lst in enumerate(lists, start=1):
        for val in lst:
            if val not in value_locations:
                value_locations[val] = []
            value_locations[val].append(i)

    # Nur Werte behalten, die in mehreren Listen vorkommen
    shared_values = {val: locs for val, locs in value_locations.items() if len(locs) > 1}

    return shared_values


def print_shared_values():
    shared = find_shared_values_with_locations()
    if shared:
        print("Gemeinsame Werte über mehrere Listen hinweg:\n")
        for val, locations in shared.items():
            loc_str = ", ".join(f"Liste {i}" for i in locations)
            print(f"Wert {val} kommt vor in: {loc_str}")
    else:
        print("Keine gemeinsamen Werte gefunden.")

