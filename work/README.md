To run application, run three commands:

pip install -r requirements.txt
cp ./privacy_news_algorithm/.env.example .env
python -m streamlit run app.py --server.port 8501 --server.address 0.0.0.0
