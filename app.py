import matplotlib.pyplot as plt
import io
import matplotlib
import pandas as pd
matplotlib.use('Agg')
import numpy as np
from flask import Flask, render_template_string, request
from flask_cors import CORS
from flask import jsonify

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:8080"}})  # Enable CORS for specific origin

def compute_graph_data(param1=1.0, param2=0.0):
    x = np.linspace(0, 10, 100)
    y = np.sin(x * param1 + param2)
    return x, y

@app.route('/compute', methods=['POST'])
def compute():
    params = request.get_json()
    x, y = compute_graph_data(params.get("param1"), params.get("param2"))
    graph_html = render_graph(x, y)
    return jsonify({'graph_html': graph_html})

def render_graph(x, y):
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set(title="Graph", xlabel="X-axis", ylabel="Y-axis")
    buf = io.BytesIO()
    plt.savefig(buf, format="svg")
    svg_data = buf.getvalue().decode()
    buf.close()
    plt.close(fig)

    # Add width and height attributes to the SVG data
    width = 500
    height = 350
    svg_data = svg_data.replace('<svg ', f'<svg width="{width}" height="auto" ')
    
    return f'''
    <!DOCTYPE html>
    <html>
    <body>
    <div>
        {svg_data}
    </div>
    </body>
    </html>
    '''

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and file.filename.endswith('.csv'):
        df = pd.read_csv(file)
        param1 = float(df['param1'].iloc[0]) if 'param1' in df.columns else 1.0
        param2 = float(df['param2'].iloc[0]) if 'param2' in df.columns else 0.0
        x, y = compute_graph_data(param1, param2)
        graph_html = render_graph(x, y)
        return jsonify({'param1': param1, 'param2': param2, 'graph_html': graph_html})

    return jsonify({'error': 'Invalid file format'}), 400

if __name__ == '__main__':
    app.run(debug=True)