import json
import os
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
import openai

# Use a non-GUI backend for Matplotlib to avoid macOS NSWindow issues
matplotlib.use("Agg")

# Try importing cuGraph for GPU acceleration
try:
    import cudf
    import cugraph as cg
    USE_CUGRAPH = True
except ImportError:
    print("cuGraph not found. Falling back to NetworkX (CPU-based).")
    USE_CUGRAPH = False

# Initialize Flask App
app = Flask(__name__)

openai.api_key = "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"


# Global variable for the graph
G = None

# Paths
UPLOAD_FOLDER = "uploads"
GRAPH_IMAGE_PATH = "static/graph.png"

# Ensure the upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("static", exist_ok=True)


def load_json(file_path):
    """Load and validate JSON data."""
    try:
        with open(file_path, "r") as file:
            data = json.load(file)
            if not isinstance(data, dict):
                raise ValueError("Invalid JSON format: Expected a dictionary.")
            return data
    except (json.JSONDecodeError, FileNotFoundError, ValueError) as e:
        print(f"Error loading JSON: {e}")
        return None


def create_graph(data):
    """Create a directed graph using NetworkX or cuGraph."""
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])

    if not isinstance(nodes, list) or not isinstance(edges, list):
        print("Invalid JSON structure: 'nodes' and 'edges' should be lists.")
        return None

    if USE_CUGRAPH:
        print("Using NVIDIA cuGraph for GPU acceleration!")
        df_edges = cudf.DataFrame({"source": [edge["source"] for edge in edges], "target": [edge["target"] for edge in edges]})
        G = cg.DiGraph()
        G.from_cudf_edgelist(df_edges, source="source", destination="target")
    else:
        print("Using NetworkX (CPU-based graph processing).")
        G = nx.DiGraph()
        for node in nodes:
            node_id = node.get("id")
            if node_id:
                G.add_node(node_id, **node)

        for edge in edges:
            source, target = edge.get("source"), edge.get("target")
            if source and target:
                G.add_edge(source, target, **edge)

    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G


def visualize_graph(G):
    """Visualize the graph and save it as an image."""
    if G is None:
        print("Graph visualization skipped due to errors.")
        return None

    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="gray",
            node_size=2000, font_size=10, font_weight="bold", arrowsize=15)

    plt.savefig(GRAPH_IMAGE_PATH)
    plt.close()

    print(f"Graph saved at {GRAPH_IMAGE_PATH}")
    return GRAPH_IMAGE_PATH


@app.route("/")
def home():
    """Render the upload page."""
    return render_template("index.html")


@app.route("/graph")
def graph_page():
    """Render the graph visualization page."""
    return render_template("graph.html")


@app.route("/upload", methods=["POST"])
def upload_json():
    """Upload JSON file, create graph, and redirect to visualization page."""
    global G

    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    file_path = os.path.join(UPLOAD_FOLDER, "uploaded_dataset.json")
    file.save(file_path)

    data = load_json(file_path)
    if data is None:
        return jsonify({"error": "Invalid JSON format"}), 400

    G = create_graph(data)
    if G is None:
        return jsonify({"error": "Graph creation failed"}), 500

    visualize_graph(G)
    
    return redirect(url_for("graph_page"))


@app.route("/graph-image")
def get_graph_image():
    """Serve the generated graph image."""
    if not os.path.exists(GRAPH_IMAGE_PATH):
        return jsonify({"error": "Graph image not found. Please upload a JSON file first."}), 404

    return send_file(GRAPH_IMAGE_PATH, mimetype="image/png")


if __name__ == "__main__":
    try:
        app.run(debug=True, port=5000)
    except SystemExit:
        print("Flask exited unexpectedly. Ignoring SystemExit exception.")
