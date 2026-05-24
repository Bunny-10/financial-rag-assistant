"""
S&P 500 company directory.
Primary source: Wikipedia's live constituent table.
Falls back to a hardcoded top-100 list if the fetch fails.
"""
import requests
from bs4 import BeautifulSoup

_cache: dict[str, str] | None = None

_FALLBACK: dict[str, str] = {
    "AAPL": "Apple Inc.", "MSFT": "Microsoft Corporation", "NVDA": "NVIDIA Corporation",
    "GOOGL": "Alphabet Inc. (Class A)", "GOOG": "Alphabet Inc. (Class C)",
    "AMZN": "Amazon.com Inc.", "META": "Meta Platforms Inc.", "TSLA": "Tesla Inc.",
    "BRK-B": "Berkshire Hathaway Inc.", "AVGO": "Broadcom Inc.",
    "JPM": "JPMorgan Chase & Co.", "LLY": "Eli Lilly and Company",
    "V": "Visa Inc.", "UNH": "UnitedHealth Group Inc.", "XOM": "Exxon Mobil Corporation",
    "MA": "Mastercard Inc.", "JNJ": "Johnson & Johnson", "PG": "Procter & Gamble Co.",
    "HD": "Home Depot Inc.", "COST": "Costco Wholesale Corporation",
    "WMT": "Walmart Inc.", "NFLX": "Netflix Inc.", "CRM": "Salesforce Inc.",
    "BAC": "Bank of America Corporation", "CVX": "Chevron Corporation",
    "MRK": "Merck & Co. Inc.", "ABBV": "AbbVie Inc.", "AMD": "Advanced Micro Devices Inc.",
    "KO": "Coca-Cola Company", "PEP": "PepsiCo Inc.", "TMO": "Thermo Fisher Scientific Inc.",
    "ORCL": "Oracle Corporation", "WFC": "Wells Fargo & Company",
    "ADBE": "Adobe Inc.", "MCD": "McDonald's Corporation", "ACN": "Accenture plc",
    "CSCO": "Cisco Systems Inc.", "LIN": "Linde plc", "ABT": "Abbott Laboratories",
    "INTC": "Intel Corporation", "GE": "GE Aerospace", "TXN": "Texas Instruments Inc.",
    "IBM": "International Business Machines Corporation", "NOW": "ServiceNow Inc.",
    "INTU": "Intuit Inc.", "CAT": "Caterpillar Inc.", "RTX": "RTX Corporation",
    "QCOM": "Qualcomm Inc.", "AMGN": "Amgen Inc.", "GS": "Goldman Sachs Group Inc.",
    "SPGI": "S&P Global Inc.", "DHR": "Danaher Corporation", "AMAT": "Applied Materials Inc.",
    "MS": "Morgan Stanley", "AXP": "American Express Company", "T": "AT&T Inc.",
    "ISRG": "Intuitive Surgical Inc.", "PFE": "Pfizer Inc.", "NEE": "NextEra Energy Inc.",
    "BKNG": "Booking Holdings Inc.", "UNP": "Union Pacific Corporation",
    "SYK": "Stryker Corporation", "VRTX": "Vertex Pharmaceuticals Inc.",
    "LOW": "Lowe's Companies Inc.", "C": "Citigroup Inc.", "ETN": "Eaton Corporation plc",
    "BSX": "Boston Scientific Corporation", "MU": "Micron Technology Inc.",
    "ADP": "Automatic Data Processing Inc.", "DE": "Deere & Company",
    "LRCX": "Lam Research Corporation", "PLD": "Prologis Inc.",
    "TJX": "TJX Companies Inc.", "MDLZ": "Mondelez International Inc.",
    "ADI": "Analog Devices Inc.", "REGN": "Regeneron Pharmaceuticals Inc.",
    "KLAC": "KLA Corporation", "CI": "The Cigna Group", "CME": "CME Group Inc.",
    "MMC": "Marsh & McLennan Companies Inc.", "PANW": "Palo Alto Networks Inc.",
    "ZTS": "Zoetis Inc.", "CB": "Chubb Limited", "SO": "Southern Company",
    "BMY": "Bristol-Myers Squibb Company", "DUK": "Duke Energy Corporation",
    "SCHW": "Charles Schwab Corporation", "BX": "Blackstone Inc.",
    "ICE": "Intercontinental Exchange Inc.", "SHW": "Sherwin-Williams Company",
    "EQIX": "Equinix Inc.", "AON": "Aon plc", "PH": "Parker-Hannifin Corporation",
    "MCO": "Moody's Corporation", "USB": "U.S. Bancorp",
    "SNPS": "Synopsys Inc.", "CDNS": "Cadence Design Systems Inc.",
    "MCK": "McKesson Corporation", "WM": "Waste Management Inc.",
    "ITW": "Illinois Tool Works Inc.", "EMR": "Emerson Electric Co.",
    "COF": "Capital One Financial Corporation", "GEV": "GE Vernova Inc.",
    "PYPL": "PayPal Holdings Inc.", "CL": "Colgate-Palmolive Company",
}


def get_sp500() -> dict[str, str]:
    """Return {ticker: company_name} for S&P 500, sorted by company name."""
    global _cache
    if _cache is not None:
        return _cache
    try:
        resp = requests.get(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            timeout=10,
            headers={"User-Agent": "FinancialRAGAssistant research@example.com"},
        )
        soup = BeautifulSoup(resp.text, "html.parser")
        table = soup.find("table", {"id": "constituents"})
        result: dict[str, str] = {}
        for row in table.find_all("tr")[1:]:
            cols = row.find_all("td")
            if len(cols) >= 2:
                ticker = cols[0].text.strip().replace(".", "-")
                company = cols[1].text.strip()
                result[ticker] = company
        if result:
            _cache = dict(sorted(result.items(), key=lambda x: x[1]))
            return _cache
    except Exception:
        pass
    _cache = dict(sorted(_FALLBACK.items(), key=lambda x: x[1]))
    return _cache


def display_options(companies: dict[str, str]) -> list[str]:
    """Return ['Company Name (TICKER)', ...] sorted by company name."""
    return [f"{name} ({ticker})" for ticker, name in companies.items()]


def parse_selection(selection: str) -> tuple[str, str]:
    """Parse 'Company Name (TICKER)' back to (ticker, company_name)."""
    ticker = selection.rsplit("(", 1)[-1].rstrip(")")
    company = selection.rsplit("(", 1)[0].strip()
    return ticker, company
