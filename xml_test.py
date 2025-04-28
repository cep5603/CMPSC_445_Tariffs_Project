import xml.etree.ElementTree as ET
import pandas as pd

def parse_wits_xml(fn: str) -> pd.Series:
    """
    Parse a WITS SDMX XML file with one <Series> of <Obs> entries,
    extracting TIME_PERIOD and OBS_VALUE into a pandas Series.
    """
    tree = ET.parse(fn)
    root = tree.getroot()
    # find all <Obs> elements under any namespace
    obs = root.findall(".//{*}Obs")
    data = []
    for o in obs:
        year = o.attrib["TIME_PERIOD"]
        val  = float(o.attrib["OBS_VALUE"])
        data.append((int(year), val))
    # build Series indexed by datetime
    df = pd.DataFrame(data, columns=["year", "tariff_rate"])
    df["date"] = pd.to_datetime(df["year"], format="%Y")
    return df.set_index("date")["tariff_rate"].sort_index()

# Usage
steel_ingot_tariffs = parse_wits_xml("tariffs_720610.xml")
print(steel_ingot_tariffs)
