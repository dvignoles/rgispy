"""Domain specific utility functions"""


def usa_classify_division(st):
    st = st.strip().upper()
    if st in ["WA", "OR", "CA", "AK", "HI"]:
        return "Pacific West"
    elif st in ["MT", "ID", "WY", "NV", "UT", "CO", "AZ", "NM"]:
        return "Mountain West"
    elif st in ["ND", "MN", "SD", "NE", "IA", "KS", "MO"]:
        return "West North Central"
    elif st in ["WI", "MI", "IL", "IN", "OH"]:
        return "East North Central"
    elif st in ["OK", "AR", "TX", "LA"]:
        return "West South Central"
    elif st in ["KY", "TN", "MS", "AL"]:
        return "East South Central"
    elif st in ["WV", "MD", "DE", "DC", "VA", "NC", "SC", "GA", "FL"]:
        return "South Atlantic"
    elif st in [
        "ME",
        "VT",
        "NH",
        "MA",
        "CT",
        "RI",
    ]:
        return "New England"
    elif st in ["NY", "PA", "NJ"]:
        return "Middle Atlantic"


def usa_classify_region(st):

    division = usa_classify_division(st)
    if division in ["Pacific West", "Mountain West"]:
        return "West"
    elif division in ["West North Central", "East North Central"]:
        return "MidWest"
    elif division in ["West South Central", "East South Central", "South Atlantic"]:
        return "South"
    elif division in ["New England", "Middle Atlantic"]:
        return "NorthEast"
