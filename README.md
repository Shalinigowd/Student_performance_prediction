Student Performance Predictor (with IQ callback integration)
============================================================

This project contains an option-2 style integration for an IQ test: the IQ site redirects
to /iq-callback?token=... and the Flask backend fetches the IQ from a provider API.

How to run locally (quick):
1. Create & activate virtualenv
2. pip install -r requirements.txt
3. Run iq_service for testing: python iq_service/app.py
4. Train model: python train_model.py
5. Run app: python app.py
6. Open http://localhost:8080
