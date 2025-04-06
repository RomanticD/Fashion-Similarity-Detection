from flask import Blueprint, request, jsonify
from ..auth.supabase_client import SupabaseClient
import json
from functools import wraps

# Create a Blueprint for auth routes
api_auth = Blueprint('auth', __name__, url_prefix='/api/auth')

# Initialize the Supabase client
supabase = SupabaseClient()


# Authentication middleware
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization')

        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({
                'success': False,
                'message': 'Authorization header is required',
                'data': None
            }), 401

        return f(*args, **kwargs)

    return decorated


def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization')

        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({
                'success': False,
                'message': 'Authorization header is required',
                'data': None
            }), 401

        try:
            # Get current user
            user_response = supabase.get_user()

            if not user_response or not user_response.user:
                return jsonify({
                    'success': False,
                    'message': 'User not found or not authenticated',
                    'data': None
                }), 401

            # Check if user is admin
            user_role = supabase.get_user_role(user_response.user.id)

            if user_role != 'admin':
                return jsonify({
                    'success': False,
                    'message': 'Admin privileges required',
                    'data': None
                }), 403

            return f(*args, **kwargs)
        except Exception as e:
            return jsonify({
                'success': False,
                'message': str(e),
                'data': None
            }), 500

    return decorated


@api_auth.route('/register', methods=['POST'])
def register():
    """
    Register a new user
    """
    try:
        data = request.json

        # Validate required fields
        if not data or 'email' not in data or 'password' not in data:
            return jsonify({
                'success': False,
                'message': 'Email and password are required',
                'data': None
            }), 400

        # Get data from request
        email = data.get('email')
        password = data.get('password')
        is_admin = data.get('is_admin', False)

        # Register the user
        response = supabase.sign_up(email, password, is_admin)

        # Return success response
        return jsonify({
            'success': True,
            'message': 'User registered successfully',
            'data': {
                'user_id': response.user.id,
                'email': response.user.email
            }
        }), 201

    except Exception as e:
        # Return error response
        return jsonify({
            'success': False,
            'message': str(e),
            'data': None
        }), 500


@api_auth.route('/login', methods=['POST'])
def login():
    """
    Login an existing user
    """
    try:
        data = request.json

        # Validate required fields
        if not data or 'email' not in data or 'password' not in data:
            return jsonify({
                'success': False,
                'message': 'Email and password are required',
                'data': None
            }), 400

        # Get data from request
        email = data.get('email')
        password = data.get('password')

        # Login the user
        response = supabase.sign_in(email, password)

        # Get the user role
        user_role = supabase.get_user_role(response.user.id)

        # Return success response
        return jsonify({
            'success': True,
            'message': 'User logged in successfully',
            'data': {
                'user_id': response.user.id,
                'email': response.user.email,
                'role': user_role,
                'access_token': response.session.access_token,
                'refresh_token': response.session.refresh_token
            }
        }), 200

    except Exception as e:
        # Return error response
        return jsonify({
            'success': False,
            'message': str(e),
            'data': None
        }), 500


@api_auth.route('/logout', methods=['POST'])
def logout():
    """
    Logout the current user
    """
    try:
        # Logout the user
        supabase.sign_out()

        # Return success response
        return jsonify({
            'success': True,
            'message': 'User logged out successfully',
            'data': None
        }), 200

    except Exception as e:
        # Return error response
        return jsonify({
            'success': False,
            'message': str(e),
            'data': None
        }), 500


@api_auth.route('/user', methods=['GET'])
@token_required
def get_user():
    """
    Get the current user information
    """
    try:

        # Get current user
        user_response = supabase.get_user()

        if not user_response or not user_response.user:
            return jsonify({
                'success': False,
                'message': 'User not found or not authenticated',
                'data': None
            }), 404

        # Get user role
        user_role = supabase.get_user_role(user_response.user.id)

        # Return success response
        return jsonify({
            'success': True,
            'message': 'User fetched successfully',
            'data': {
                'user_id': user_response.user.id,
                'email': user_response.user.email,
                'role': user_role,
                'is_admin': user_role == 'admin'
            }
        }), 200

    except Exception as e:
        # Return error response
        return jsonify({
            'success': False,
            'message': str(e),
            'data': None
        }), 500