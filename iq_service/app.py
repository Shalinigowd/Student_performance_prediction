from flask import Flask, jsonify, request
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
IQ_STORE = {'testtoken': 119, 'user123': 104}

@app.route('/api/iq')
def api_iq():
    token = request.args.get('token')
    if not token:
        return jsonify({'error':'missing token'}), 400
    iq = IQ_STORE.get(token)
    if iq is None:
        return jsonify({'error':'not found'}), 404
    return jsonify({'iq': iq})

@app.route('/iqtest')
def iq_test_page():
    # simple HTML simulate test finish with redirect to callback token
    return '''
    <h2>Simulated IQ Test</h2>
    <p>Click finish to redirect to the main app with token "testtoken".</p>
    <button onclick="window.location='http://localhost:8080/iq-callback?token=testtoken'">Finish test</button>
    '''

if __name__ == '__main__':
    app.run(port=5001, debug=True)
