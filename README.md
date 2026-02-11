# How to Run Locally

### For first time users, run the following commands in order {.tabset}
### Tab 1 
pip install -r requirements.txt
### Tab 2
cp ./privacy_news_algorithm/.env.example .env
python -m streamlit run app.py --server.port 8501 --server.address 0.0.0.0
