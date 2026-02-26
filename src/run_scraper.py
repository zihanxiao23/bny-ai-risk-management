from yfinance_scraper import YahooScraper
from gnews_scraper import GNewsScraper

ENTITIES = {
    "EURUSD": ["EURUSD=X"],
    "GBPUSD": ["GBPUSD=X"],
    "USDJPY": ["USDJPY=X"],
    "DXY": ["DX-Y.NYB", "UUP"]
}
def scrape_all():
    try:
        yfinance = YahooScraper(ENTITIES)
        yfinance.fetch().save_csv("eiei2222222.csv")
    except Exception as e:
        print(f"Critical Failure in Yahoo Pipeline: {e}")

    # try:
    #     gnews = GNewsScraper(entities={},topics=EXPAND_TOPICS,start_date="2025-01-01", end_date="2025-12-31")
    #     # gnews.fetch().save()
    #     gnews.eiei()
    # except Exception as e:
    #     print(f"Critical Failure in Gnews Pipeline: {e}")

if __name__ == "__main__":
    scrape_all()