from flask import Flask, jsonify, Blueprint
from flask_cors import cross_origin, CORS


from src.app.supabse_route import token_required
from src.repo.history_repo import read_search_history



api_sh = Blueprint('search_history', __name__)
CORS(api_sh)

@api_sh.route('/search_history/<search_id>', methods=['GET'])
@cross_origin()
@token_required
def get_search_history(search_id):
    success = read_search_history(search_id)
    if success:
        return jsonify({"success": True, "message": success})
    else:
        return jsonify({"success": False, "message": "查询记录失败，检查后端日志"}), 500

