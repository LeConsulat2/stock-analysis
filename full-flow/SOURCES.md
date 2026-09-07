# Sources and date-sensitive configuration

Checked 2026-09-08 for this teaching example. Sources below support configuration or implementation, not current buy/sell opinions. Local yfinance calls and live model calls were not made.

## Models and tools

- [OpenAI GPT-6 Astra](https://developers.openai.com/api/docs/models/gpt-6-astra): API ID and model capabilities; account access remains untested.
- [OpenAI function calling](https://developers.openai.com/api/docs/guides/function-calling): retain assistant output items and return correlated function results. The adapter retains encrypted reasoning items for stateless continuation.
- [Claude model overview](https://platform.claude.com/docs/en/models/overview): the fetched page lists `claude-fable-5-1`; a product family name alone is not always an API ID.
- [Claude tool use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview): native tool-use/result turns and complete content preservation.
- [Gemini current model guide](https://ai.google.dev/gemini-api/docs/latest-model): the fetched page lists `gemini-3.8-flash`. Search snippets had older versions; configuration follows the fetched page.
- [Gemini function calling](https://ai.google.dev/gemini-api/docs/function-calling): manual function calling through the native SDK; preserve returned model content and any thought signatures.
- [yfinance Search](https://ranaroussi.github.io/yfinance/reference/api/yfinance.Search.html): bounded query/news parameters. Search results are incomplete news coverage, not a general research engine.
- [yfinance documentation](https://ranaroussi.github.io/yfinance/): unofficial research/personal-use data access; check terms before broader distribution.

## Companies, brands and market hours

- [Acushnet investor relations](https://www.acushnetholdingscorp.com/overview/default.aspx) and [brands](https://www.acushnetholdingscorp.com/our-brands/default.aspx): GOLF and the Titleist/FootJoy brand grouping.
- [Callaway investor relations](https://ir.callawaygolf.com/investor-overview): the current page identifies NYSE CALY and Callaway, Odyssey, TravisMathew and OGIO. Historical MODG references need reassessment.
- [SpaceX IPO pricing announcement, June 11, 2026](https://ir.spacex.com/updates/releases-details/2026/Space-Exploration-Technologies-Corp--Announces-Pricing-of-Initial-Public-Offering/default.aspx): identifies SPCX and expected June 12 trading. This supports the configured candidate ticker, not an assumption of Yahoo coverage or unchecked instrument identity.
- [PGA TOUR Enterprises announcement](https://pgatourmedia.pgatourhq.com/static-assets/page/files/pressreleases/2024/03/PGATOURSSGpartnershipannouncement.pdf): commercial venture and strategic investment/player equity structure. No direct listed ticker was established in this review.
- [LIV Golf investor transition](https://www.livgolf.com/news/liv-golf-reaches-agreement-with-lead-investor-for-its-next-era): financing and ownership context can change; no direct listed ticker established.
- [TMRW Sports Series A announcement](https://tmrwsportsgroup.com/articles/2024/06/tmrw-sports-completes-series-a-investment-round%2C-co-led-by-dynasty-equity-and-connect-ventures): private investment context for TMRW/TGL, not a direct retail stock listing.
- [PUMA golf](https://about.puma.com/en/this-is-puma/our-sports/golf): another golf-related brand group to investigate separately from the initial watchlist.
- [NYSE trading hours](https://www.nyse.com/trade/hours-calendars): regular US equity session ends at 16:00 Eastern. Auckland conversion is calculated with timezone rules rather than a fixed UTC offset.

Other familiar ticker mappings in `watchlist.py` are validated against returned equity/name/currency metadata before use. Configured related-company links are hypotheses; no current supply contract, ownership stake or revenue exposure is asserted by that list.
