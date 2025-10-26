app = Flask(__name__, static_folder='static', template_folder='static')


@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.json
    text = data.get('text', '')
    if not text:
        return jsonify({'error': 'no text provided'}), 400
    try:
        result = predict(text)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
