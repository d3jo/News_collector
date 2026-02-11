# 🛡️ Privacy News Monitor

> Automated 48-hour privacy news aggregation system  
> Built for a research startup to streamline daily monitoring

---

## 🌐 Live Application

🔗 **Try it here:**  
https://news-collector.streamlit.app/

---

## About the News Collector

This application automatically retrieves privacy-related news published within the last **48 hours**.

Originally developed as outsourced work for a research startup that required daily monitoring of privacy-focused news.

### Before Automation
- Manual collection process
- ~3 hours per day
- Required active searching and filtering

### After Automation
- Fully automated pipeline
- Under 3 minutes runtime
- Significant time savings
- Streamlined daily workflow

---

## How it works

It uses 6 news source APIs to fetch the hottest news on web. Namely, NewsAPI, MediaStack, Currents, etc.

It uses strict filtering algorithm on the 48-hour window fetched news articles, then filter duplicated news from there because same news from different sources are collected very often.

After filtering, it picks the top 20 articles from there. The newses are requested to be categorized into 4 topics, so it focuses on the newest articles, even distribution of categories, duplicate avoidance.

OpenAI API was used to summarize each 20 articles in a 3-4 bullet point which is the core automation process they requested for.
