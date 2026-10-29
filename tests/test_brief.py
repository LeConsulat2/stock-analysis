"""Offline tests for brief.py: calculations, flags, concepts and rendering, no network."""

from __future__ import annotations

import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import brief  # noqa: E402


def closes(values: list[float], start: str = "2026-01-01") -> pd.Series:
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq="B"))


def priced(ticker: str, name: str, **fields) -> brief.StockLine:
    line = brief.StockLine(ticker=ticker, name=name, currency=fields.pop("currency", "USD"), price=100.0)
    for key, value in fields.items():
        setattr(line, key, value)
    return line


class DescribeClosesTests(unittest.TestCase):
    def test_basic_numbers(self) -> None:
        values = [100.0] * 30 + [110.0]
        stats = brief.describe_closes(closes(values))
        self.assertAlmostEqual(stats["day_pct"], 10.0)
        self.assertAlmostEqual(stats["year_pct"], 10.0)
        self.assertAlmostEqual(stats["month_pct"], 10.0)
        self.assertAlmostEqual(stats["from_high_pct"], 0.0)
        self.assertAlmostEqual(stats["from_low_pct"], 10.0)
        self.assertEqual(stats["asof"], closes(values).index[-1].date())
        self.assertIsNotNone(stats["typical_move_pct"])

    def test_too_little_data_returns_empty(self) -> None:
        self.assertEqual(brief.describe_closes(closes([100.0])), {})
        self.assertEqual(brief.describe_closes(closes([100.0, float("nan")])), {})

    def test_short_history_has_no_typical_move(self) -> None:
        self.assertIsNone(brief.describe_closes(closes([100.0, 101.0, 102.0]))["typical_move_pct"])

    def test_pct_change_guards_zero_and_missing(self) -> None:
        self.assertIsNone(brief.pct_change(1.0, 0))
        self.assertIsNone(brief.pct_change(None, 1.0))
        self.assertIsNone(brief.pct_change(1.0, float("nan")))


class HomeCurrencyTests(unittest.TestCase):
    def test_weaker_nzd_raises_nzd_return(self) -> None:
        # +10% in USD while NZDUSD falls 0.60 -> 0.50: worth 1.1 * 1.2 = +32% in NZD.
        self.assertAlmostEqual(brief.to_home_return(10.0, 0.60, 0.50), 32.0)

    def test_stronger_nzd_lowers_nzd_return(self) -> None:
        self.assertLess(brief.to_home_return(10.0, 0.50, 0.60), 10.0)

    def test_missing_inputs(self) -> None:
        self.assertIsNone(brief.to_home_return(None, 0.6, 0.5))
        self.assertIsNone(brief.to_home_return(10.0, 0.6, 0))


class FlagTests(unittest.TestCase):
    today = date(2026, 10, 29)

    def test_big_day_is_relative_to_typical_move(self) -> None:
        calm = priced("KO", "Coca-Cola", day_pct=3.0, typical_move_pct=1.0, from_high_pct=-5, from_low_pct=5)
        wild = priced("TSLA", "Tesla", day_pct=3.0, typical_move_pct=3.5, from_high_pct=-5, from_low_pct=5)
        self.assertTrue(any("big day" in f for f in brief.find_flags(calm, self.today)))
        self.assertEqual(brief.find_flags(wild, self.today), [])

    def test_new_high_and_low(self) -> None:
        line = priced("AAPL", "Apple", day_pct=0.1, typical_move_pct=1.5, from_high_pct=0.0, from_low_pct=30)
        self.assertIn("closed at a new 1-year high", brief.find_flags(line, self.today))
        line = priced("AAPL", "Apple", day_pct=0.1, typical_move_pct=1.5, from_high_pct=-30, from_low_pct=0.0)
        self.assertIn("closed at a new 1-year low", brief.find_flags(line, self.today))

    def test_earnings_window(self) -> None:
        line = priced("GOOGL", "Alphabet", from_high_pct=-5, from_low_pct=5)
        line.next_earnings = self.today
        self.assertTrue(any("earnings today" in f for f in brief.find_flags(line, self.today)))
        line.next_earnings = date(2026, 11, 3)
        self.assertTrue(any("in 5 days" in f for f in brief.find_flags(line, self.today)))
        line.next_earnings = date(2026, 10, 28)  # already reported
        self.assertEqual(brief.find_flags(line, self.today), [])
        line.next_earnings = date(2026, 11, 20)  # too far away
        self.assertEqual(brief.find_flags(line, self.today), [])


class ConceptTests(unittest.TestCase):
    def test_benchmark_counts_winners(self) -> None:
        lines = [
            priced("VOO", "S&P 500", year_pct=15.0),
            priced("KO", "Coca-Cola", year_pct=38.0),
            priced("TSLA", "Tesla", year_pct=-12.0),
        ]
        title, text = brief.concept_benchmark(lines)
        self.assertIn("1 of 2", text)
        self.assertIn("Coca-Cola", text)

    def test_benchmark_needs_the_index(self) -> None:
        self.assertIsNone(brief.concept_benchmark([priced("KO", "Coca-Cola", year_pct=38.0)]))

    def test_errored_lines_are_ignored(self) -> None:
        bad = brief.StockLine(ticker="XXX", name="Broken", error="no data")
        self.assertIsNone(brief.concept_typical_move([bad]))

    def test_pick_concept_falls_through_to_one_that_applies(self) -> None:
        # Only the trend-split concept can be illustrated by this data.
        lines = [priced("TSLA", "Tesla", month_pct=14.0, year_pct=-12.0)]
        for day in range(1, 8):
            self.assertEqual(brief.pick_concept(lines, date(2026, 10, day))[0], "Short trend against long trend")

    def test_pick_concept_none_when_nothing_applies(self) -> None:
        self.assertIsNone(brief.pick_concept([], date(2026, 10, 29)))


class WatchlistAndRenderTests(unittest.TestCase):
    def test_load_watchlist_skips_comments_and_blank_tickers(self) -> None:
        content = '﻿# a comment\nticker,name,why,wrong_if\naapl,Apple,"Services, buybacks",\n,,,\nKO,Coca-Cola,,\n'
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "watchlist.csv"
            path.write_text(content, encoding="utf-8")
            lines = brief.load_watchlist(path)
        self.assertEqual([line.ticker for line in lines], ["AAPL", "KO"])
        self.assertEqual(lines[0].why, "Services, buybacks")

    def test_repo_watchlist_parses(self) -> None:
        lines = brief.load_watchlist()
        self.assertIn("005930.KS", [line.ticker for line in lines])
        self.assertIn(brief.BENCHMARK, [line.ticker for line in lines])

    def test_render_survives_errors_and_missing_values(self) -> None:
        ok = priced("005930.KS", "Samsung Electronics", currency="KRW", day_pct=-4.1,
                    typical_move_pct=1.9, from_high_pct=-36.0, from_low_pct=120.0)
        ok.price, ok.asof = 232000.0, date(2026, 10, 29)
        ok.flags = brief.find_flags(ok, date(2026, 10, 29))
        bad = brief.StockLine(ticker="NOPE", name="Missing", error="no recent prices returned")
        report = brief.Brief(
            today=date(2026, 10, 29),
            lines=[ok, bad],
            fx=[brief.FxLine("NZDUSD=X", rate=0.5618, day_pct=-1.09, year_pct=-2.86),
                brief.FxLine("NZDKRW=X", error="timeout")],
            concept=None,
        )
        text = brief.render_text(report)
        html = brief.render_html(report)
        self.assertIn("₩232,000", text)
        self.assertNotIn("$232,000", text)
        self.assertIn("2 to look at", text)
        self.assertIn("data problem: no recent prices returned", text)
        self.assertIn("NZD weaker this year", text)
        self.assertIn("Korea close 29 Oct", text)
        self.assertIn("₩232,000", html)
        self.assertIn("data problem", html)

    def test_quiet_day_says_so(self) -> None:
        line = priced("KO", "Coca-Cola", day_pct=0.2, typical_move_pct=1.0, from_high_pct=-4, from_low_pct=30)
        report = brief.Brief(today=date(2026, 10, 29), lines=[line], fx=[], concept=None)
        self.assertIn("nothing unusual", brief.subject_line(report))
        self.assertIn("Nothing unusual", brief.render_text(report))


if __name__ == "__main__":
    unittest.main()
