BROKERAGE_CODE_MAP = {
    "VPS": "VPS",
    "SSI": "SSI",
    "TCBS": "TCX",
    "VCI": "VIETCAP",
    "HCM": "HSC",
    "MBS": "MBS",
    "VND": "VNDS",
    "MAS": "MAS",
    "KIS": "KIS",
    "FTS": "FPTS",
    "VCBS": "VCBS",
    "VPBS": "VPBS",
}


def get_brokerage_code(ticker: str) -> str:
    if not ticker:
        return ticker
    return BROKERAGE_CODE_MAP.get(ticker.upper(), ticker.upper())
