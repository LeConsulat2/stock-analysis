"""Instrument identity is configuration, never a model's guess.

Relationships below are research leads, NOT verified supplier contracts.
Listing sources and review date are in SOURCES.md. Recheck after corporate actions.
"""

from contracts import Instrument


def us(key: str, name: str, ticker: str, identity: tuple[str, ...],
       topics: tuple[str, ...], related: tuple[str, ...] = (),
       note: str = "") -> Instrument:
    return Instrument(key, name, ticker, "USD", "America/New_York", 16,
                      identity_terms=identity, topics=topics, related=related, note=note)


WATCHLIST = (
    us("acushnet", "Acushnet (Titleist / FootJoy)", "GOLF", ("acushnet",),
       ("Titleist golf demand", "golf equipment tariffs"), ("CALY",),
       "Titleist and FootJoy are brands; GOLF is the parent equity."),
    us("callaway", "Callaway Golf", "CALY", ("callaway",),
       ("Callaway golf equipment", "golf participation"), ("GOLF",),
       "Use current CALY identity; do not assume historic MODG structure."),
    Instrument("samsung", "Samsung Electronics", "005930.KS", "KRW",
               "Asia/Seoul", 15, 30, ("samsung",),
               ("Samsung memory HBM", "semiconductor capacity supply"),
               ("NVDA", "MU", "000660.KS")),
    us("nvidia", "NVIDIA", "NVDA", ("nvidia",),
       ("NVIDIA supply chain", "AI data center capital expenditure"),
       ("TSM", "MU", "AVGO")),
    us("alphabet", "Alphabet Class A", "GOOGL", ("alphabet",),
       ("Google advertising AI search", "Google cloud capital expenditure"),
       ("NVDA", "MSFT")),
    us("apple", "Apple", "AAPL", ("apple",),
       ("Apple iPhone demand", "Apple supply chain"), ("TSM", "QCOM")),
    us("adobe", "Adobe", "ADBE", ("adobe",),
       ("Adobe creative subscriptions", "generative AI creative software"),
       ("MSFT",)),
    us("tesla", "Tesla", "TSLA", ("tesla",),
       ("Tesla deliveries margins", "electric vehicle battery pricing"),
       ("1211.HK",)),
    us("spacex", "SpaceX", "SPCX", ("space exploration", "spacex"),
       ("SpaceX Starlink launch", "satellite broadband competition"), ("RKLB",),
       "June 2026 IPO announcement names SPCX. Require runtime equity/name match."),
    Instrument("pga", "PGA TOUR / PGA TOUR Enterprises", None, None,
               "America/New_York", 16, topics=("PGA TOUR Enterprises",),
               note="Context only: no direct listed equity configured; private commercial interests differ from the TOUR."),
    Instrument("liv", "LIV Golf", None, None, "America/New_York", 16,
               topics=("LIV Golf financing sponsors",),
               note="Context only: no direct listed equity verified. Sponsorship does not imply ownership."),
    Instrument("tgl", "TGL / TMRW Sports", None, None, "America/New_York", 16,
               topics=("TGL TMRW Sports",),
               note="Context only: private investment/team interests; no direct listed equity configured."),
)


def select_watchlist(keys: list[str] | None) -> list[Instrument]:
    if not keys:
        return list(WATCHLIST)
    lookup = {value.lower(): item for item in WATCHLIST
              for value in (item.key, item.ticker) if value}
    unknown = [k for k in keys if k.lower() not in lookup]
    if unknown:
        raise ValueError("Unknown watchlist entries: " + ", ".join(unknown))
    return list(dict.fromkeys(lookup[k.lower()] for k in keys))
