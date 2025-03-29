from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/summary-statistics')
def summary_statistics():
    return render_template('summary_statistics.html')

@app.route('/correlations')
def correlations():
    return render_template('correlations.html')

@app.route('/prediction-model')
def prediction_model():
    return render_template('prediction_model.html')

if __name__ == '__main__':
    app.run(debug=True)