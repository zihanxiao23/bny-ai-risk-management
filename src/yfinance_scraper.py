from base_scraper import BaseScraper
from constant import Constant
import yfinance as yf

class YahooScraper(BaseScraper):
    def __init__(self, entities):
        super().__init__()
        self.entities = entities or {}

    def get_url(self, payload):
        url = payload.get('link')
        if not url:
            ctu = payload.get('clickThroughUrl')
            if isinstance(ctu, dict):
                url = ctu.get('url')
        if not url:
            can = payload.get('canonicalUrl')
            if isinstance(can, dict):
                url = can.get('url')
        return self.clean_url(url) if url else None

    def fetch(self):
        for name,ticker_list in self.entities.items():
            for ticker in ticker_list:
                print(f"Yahoo Finance: Searching {name} with ticker {ticker}...")
                stock = yf.Ticker(ticker)

                news_items = stock.news
                if news_items is None:
                    print(f"  [SKIP] No news returned for {ticker}")
                    continue

                for item in news_items:
                    if item is None:
                        continue
                    
                    payload = item.get('content', item)
                    if not payload or not isinstance(payload, dict):
                        continue

                    url = self.get_url(payload)
                    
                    self.batch.append({
                        Constant.URL_HASH: self.hash_url(url),
                        Constant.TITLE: payload.get('title'),
                        Constant.URL: url,
                        Constant.PUBLISHED_AT: payload.get('pubDate', self.run_ts),
                        Constant.TICKER: ticker,    
                        Constant.TOPIC: name, 
                        Constant.SOURCE: Constant.SOURCE_YFINANCE
                    })

        return self