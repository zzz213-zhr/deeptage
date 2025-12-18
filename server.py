from flask import Flask, request, jsonify, render_template
from process import run_acl_infer

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

# 推理接口
@app.route("/infer", methods=["POST"])
def infer():
    if "image" not in request.files:
        return jsonify({"error": "no image uploaded"}), 400

    image_file = request.files["image"]
    image_bytes = image_file.read()

    print("收到图片字节长度 =", len(image_bytes))

    try:
        # 调用 SSH → Atlas → ACL 推理
        result = run_acl_infer(image_bytes, T=4)
        return jsonify(result)
    except Exception as e:
        print("Infer error:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
