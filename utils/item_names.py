ITEM_ALIASES = {
    "bottle": "plastic bottle",
    "mobile": "cell phone",
    "phone": "cell phone",
    "computer": "desktop computer",
    "mouse": "computer mouse",
    "tv": "television",
    "tshirt": "cotton t-shirt",
    "t-shirt": "cotton t-shirt",
}

PERSON_NAMES = frozenset({"person", "people", "human", "humans"})


def normalize_item_name(name: str) -> str:
    return " ".join(name.casefold().split())
