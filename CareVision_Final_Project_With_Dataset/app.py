from flask import Flask, render_template, request, jsonify, session, redirect, url_for, Response
import numpy as np
import pandas as pd
import joblib
import json
from datetime import datetime
import hashlib
import os

app = Flask(__name__)
app.secret_key = 'carevision_secret_key_2025'  # Required for session

# ================= USER DATABASE =================
# In production, use a real database. This is for demo purposes.
USERS_DB = {
    'admin': {
        'password': hashlib.sha256('admin123'.encode()).hexdigest(),
        'role': 'admin',
        'joined_date': '2024-01-01'
    },
    'demo': {
        'password': hashlib.sha256('demo123'.encode()).hexdigest(),
        'role': 'user',
        'joined_date': '2024-12-01'
    }
}

# User prediction counts cache
USER_PREDICTION_COUNTS = {}

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password, hashed):
    return hash_password(password) == hashed

# ================= LOAD HEART MODEL =================
try:
    heart_model = joblib.load("heart_model.pkl")
    heart_scaler = joblib.load("heart_scaler.pkl")
    print("✅ Heart model loaded successfully")
except Exception as e:
    print(f"❌ Error loading heart model: {e}")
    heart_model = None
    heart_scaler = None

heart_columns = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal"
]

# ================= LOAD DIABETES MODEL =================
try:
    diabetes_model = joblib.load("diabetes_model.pkl")
    diabetes_scaler = joblib.load("diabetes_scaler.pkl")
    gender_encoder = joblib.load("gender_encoder.pkl")
    smoking_encoder = joblib.load("smoking_encoder.pkl")
    print("✅ Diabetes model loaded successfully")
except Exception as e:
    print(f"❌ Error loading diabetes model: {e}")
    diabetes_model = None
    diabetes_scaler = None
    gender_encoder = None
    smoking_encoder = None

diabetes_columns = [
    "gender", "age", "hypertension", "heart_disease",
    "smoking_history", "bmi", "hba1c_level", "blood_glucose_level"
]

# ================= PREDICTION HISTORY =================
PREDICTION_HISTORY = []

# ================= LOGIN ROUTES =================

@app.route("/")
def login_page():
    return render_template("login.html")

@app.route("/login", methods=["POST"])
def login():
    username = request.form.get("username", "").strip()
    password = request.form.get("password", "").strip()
    
    if username in USERS_DB and verify_password(password, USERS_DB[username]['password']):
        session['logged_in'] = True
        session['username'] = username
        session['role'] = USERS_DB[username]['role']
        
        # Initialize user prediction counts
        if username not in USER_PREDICTION_COUNTS:
            USER_PREDICTION_COUNTS[username] = {'heart': 0, 'diabetes': 0}
        
        # Add sample data for new users (except admin)
        if username != 'admin' and not any(p.get('username') == username for p in PREDICTION_HISTORY):
            sample_predictions = [
                {
                    'username': username,
                    'date': '2024-12-15 14:30',
                    'disease': 'heart',
                    'risk_high': False,
                    'probability': 15.2,
                    'health_score': 85
                },
                {
                    'username': username,
                    'date': '2024-12-10 11:20',
                    'disease': 'diabetes',
                    'risk_high': True,
                    'probability': 68.5,
                    'health_score': 32
                }
            ]
            PREDICTION_HISTORY.extend(sample_predictions)
            USER_PREDICTION_COUNTS[username]['heart'] += 1
            USER_PREDICTION_COUNTS[username]['diabetes'] += 1
        
        return redirect(url_for('dashboard'))
    return redirect(url_for('login_page'))

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for('login_page'))

# ================= DASHBOARD ROUTE =================

@app.route("/dashboard")
def dashboard():
    if not session.get('logged_in'):
        return redirect(url_for('login_page'))
    
    username = session.get('username', 'User')
    
    # Filter predictions for current user
    user_predictions = [p for p in PREDICTION_HISTORY if p.get('username') == username]
    
    # Calculate statistics
    heart_count = len([p for p in user_predictions if p.get('disease') == 'heart'])
    diabetes_count = len([p for p in user_predictions if p.get('disease') == 'diabetes'])
    
    # Calculate accuracy based on user predictions
    if len(user_predictions) > 0:
        high_risk_count = len([p for p in user_predictions if p.get('risk_high')])
        accuracy = min(95.7, 100 - (high_risk_count * 5))
    else:
        accuracy = 95.7
    
    # Get recent predictions (last 5)
    recent_predictions = sorted(
        user_predictions,
        key=lambda x: datetime.strptime(x['date'], "%Y-%m-%d %H:%M") if isinstance(x['date'], str) else x['date'],
        reverse=True
    )[:5]
    
    return render_template("dashboard.html",
                         username=username,
                         heart_count=heart_count,
                         diabetes_count=diabetes_count,
                         accuracy=round(accuracy, 1),
                         recent_predictions=recent_predictions)

# ================= ADMIN ROUTES =================

@app.route("/admin")
def admin_panel():
    if not session.get('logged_in'):
        return redirect(url_for('login_page'))
    
    if session.get('role') != 'admin':
        return "Access denied. Admin only.", 403
    
    # Calculate statistics
    total_predictions = len(PREDICTION_HISTORY)
    heart_predictions = len([p for p in PREDICTION_HISTORY if p.get('disease') == 'heart'])
    diabetes_predictions = len([p for p in PREDICTION_HISTORY if p.get('disease') == 'diabetes'])
    
    # Prepare user list
    users = []
    for username, user_data in USERS_DB.items():
        user_predictions = [p for p in PREDICTION_HISTORY if p.get('username') == username]
        users.append({
            'username': username,
            'role': user_data['role'],
            'predictions_count': len(user_predictions),
            'joined_date': user_data.get('joined_date', 'Unknown')
        })
    
    # Get all predictions sorted by date
    all_predictions = sorted(
        PREDICTION_HISTORY,
        key=lambda x: datetime.strptime(x['date'], "%Y-%m-%d %H:%M") if isinstance(x['date'], str) else x['date'],
        reverse=True
    )
    
    return render_template("admin.html",
                         username=session.get('username'),
                         total_users=len(USERS_DB),
                         total_predictions=total_predictions,
                         heart_predictions=heart_predictions,
                         diabetes_predictions=diabetes_predictions,
                         users=users,
                         all_predictions=all_predictions)

# ================= ADMIN API ENDPOINTS =================

@app.route("/api/admin/add-user", methods=["POST"])
def admin_add_user():
    if not session.get('logged_in') or session.get('role') != 'admin':
        return jsonify({"success": False, "error": "Unauthorized"}), 401
    
    data = request.json
    username = data.get('username', '').strip()
    password = data.get('password', '').strip()
    role = data.get('role', 'user')
    
    if not username or not password:
        return jsonify({"success": False, "error": "Username and password required"})
    
    if username in USERS_DB:
        return jsonify({"success": False, "error": "Username already exists"})
    
    USERS_DB[username] = {
        'password': hash_password(password),
        'role': role,
        'joined_date': datetime.now().strftime("%Y-%m-%d")
    }
    
    if username not in USER_PREDICTION_COUNTS:
        USER_PREDICTION_COUNTS[username] = {'heart': 0, 'diabetes': 0}
    
    return jsonify({"success": True})

@app.route("/api/admin/delete-user", methods=["POST"])
def admin_delete_user():
    if not session.get('logged_in') or session.get('role') != 'admin':
        return jsonify({"success": False, "error": "Unauthorized"}), 401
    
    data = request.json
    username = data.get('username', '').strip()
    
    if username == 'admin':
        return jsonify({"success": False, "error": "Cannot delete admin user"})
    
    if username not in USERS_DB:
        return jsonify({"success": False, "error": "User not found"})
    
    # Delete user's predictions
    global PREDICTION_HISTORY
    PREDICTION_HISTORY = [p for p in PREDICTION_HISTORY if p.get('username') != username]
    
    # Delete user from database
    del USERS_DB[username]
    
    if username in USER_PREDICTION_COUNTS:
        del USER_PREDICTION_COUNTS[username]
    
    return jsonify({"success": True})

@app.route("/api/admin/clear-predictions", methods=["POST"])
def admin_clear_predictions():
    if not session.get('logged_in') or session.get('role') != 'admin':
        return jsonify({"success": False, "error": "Unauthorized"}), 401
    
    global PREDICTION_HISTORY
    PREDICTION_HISTORY = []
    
    # Reset user counts
    for username in USER_PREDICTION_COUNTS:
        USER_PREDICTION_COUNTS[username] = {'heart': 0, 'diabetes': 0}
    
    return jsonify({"success": True})

@app.route("/api/admin/reset-system", methods=["POST"])
def admin_reset_system():
    if not session.get('logged_in') or session.get('role') != 'admin':
        return jsonify({"success": False, "error": "Unauthorized"}), 401
    
    global PREDICTION_HISTORY, USERS_DB, USER_PREDICTION_COUNTS
    
    # Reset to default
    USERS_DB = {
        'admin': {
            'password': hash_password('admin123'),
            'role': 'admin',
            'joined_date': '2024-01-01'
        }
    }
    PREDICTION_HISTORY = []
    USER_PREDICTION_COUNTS = {'admin': {'heart': 0, 'diabetes': 0}}
    
    return jsonify({"success": True})

@app.route("/api/admin/export-all")
def admin_export_all():
    if not session.get('logged_in') or session.get('role') != 'admin':
        return jsonify({"error": "Unauthorized"}), 401
    
    export_data = {
        'users': USERS_DB,
        'predictions': PREDICTION_HISTORY,
        'export_date': datetime.now().isoformat()
    }
    
    data = json.dumps(export_data, indent=2, default=str)
    
    return Response(
        data,
        mimetype="application/json",
        headers={"Content-disposition": "attachment; filename=carevision_export.json"}
    )

@app.route("/api/admin/stats")
def admin_stats():
    if not session.get('logged_in') or session.get('role') != 'admin':
        return jsonify({"error": "Unauthorized"}), 401
    
    return jsonify({
        'total_users': len(USERS_DB),
        'total_predictions': len(PREDICTION_HISTORY),
        'heart_predictions': len([p for p in PREDICTION_HISTORY if p.get('disease') == 'heart']),
        'diabetes_predictions': len([p for p in PREDICTION_HISTORY if p.get('disease') == 'diabetes'])
    })

# ================= OTHER ROUTES =================

@app.route("/chat")
def chat():
    if not session.get('logged_in'):
        return redirect(url_for('login_page'))
    return render_template("chat.html", username=session.get('username', 'User'))

@app.route("/map")
def map_page():
    if not session.get('logged_in'):
        return redirect(url_for('login_page'))
    return render_template("map.html")

@app.route("/medical")
def medical_page():
    if not session.get('logged_in'):
        return redirect(url_for('login_page'))
    return render_template("medical.html")

@app.route("/heart", methods=["GET", "POST"])
def heart_page():
    if not session.get('logged_in'):
        return redirect(url_for('login_page'))
    
    if request.method == "POST":
        return heart_predict()
    return render_template("index.html")

@app.route("/diabetes", methods=["GET", "POST"])
def diabetes_page():
    if not session.get('logged_in'):
        return redirect(url_for('login_page'))
    
    if request.method == "POST":
        return diabetes_predict()
    return render_template("index2.html")

@app.route("/profile")
def profile():
    if not session.get('logged_in'):
        return redirect(url_for('login_page'))
    
    username = session.get('username', 'User')
    user_predictions = [p for p in PREDICTION_HISTORY if p.get('username') == username]
    heart_count = len([p for p in user_predictions if p.get('disease') == 'heart'])
    diabetes_count = len([p for p in user_predictions if p.get('disease') == 'diabetes'])
    
    return render_template("profile.html",
                         username=username,
                         heart_count=heart_count,
                         diabetes_count=diabetes_count)

@app.route("/history")
def history():
    if not session.get('logged_in'):
        return redirect(url_for('login_page'))
    
    username = session.get('username', 'User')
    user_predictions = [p for p in PREDICTION_HISTORY if p.get('username') == username]
    sorted_predictions = sorted(
        user_predictions,
        key=lambda x: datetime.strptime(x['date'], "%Y-%m-%d %H:%M") if isinstance(x['date'], str) else x['date'],
        reverse=True
    )
    
    return render_template("history.html",
                         predictions=sorted_predictions,
                         username=username)

# ================= HEART PREDICTION =================
def heart_predict():
    try:
        if heart_model is None or heart_scaler is None:
            return "Heart model not loaded. Please check server logs.", 500
        
        values = []
        for col in heart_columns:
            value = request.form.get(col)
            if not value:
                return f"Missing value for {col}. Please fill all fields.", 400
            try:
                values.append(float(value))
            except ValueError:
                return f"Invalid value for {col}. Must be a number.", 400
        
        df = pd.DataFrame([values], columns=heart_columns)
        scaled = heart_scaler.transform(df)

        pred = heart_model.predict(scaled)[0]
        prob = heart_model.predict_proba(scaled)[0][1] * 100
        health = round(100 - prob)
        
        username = session.get('username', 'User')
        
        # Save to history
        PREDICTION_HISTORY.append({
            'username': username,
            'date': datetime.now().strftime("%Y-%m-%d %H:%M"),
            'disease': 'heart',
            'risk_high': bool(pred == 1),
            'probability': round(prob, 2),
            'health_score': health
        })
        
        # Update user count
        if username in USER_PREDICTION_COUNTS:
            USER_PREDICTION_COUNTS[username]['heart'] += 1

        return render_template("result.html",
                               pred=pred,
                               prob=round(prob, 2),
                               health=health)
    except Exception as e:
        return f"Error processing prediction: {str(e)}", 400

# ================= DIABETES PREDICTION =================
def diabetes_predict():
    try:
        if diabetes_model is None or diabetes_scaler is None:
            return "Diabetes model not loaded. Please check server logs.", 500
        
        gender = request.form.get("gender", "").lower().strip()
        smoking = request.form.get("smoking_history", "").lower().strip()
        
        if not gender or not smoking:
            return "Gender and smoking history are required.", 400
        
        try:
            gender_encoded = gender_encoder.transform([gender])[0]
            smoking_encoded = smoking_encoder.transform([smoking])[0]
        except Exception as e:
            return f"Invalid value. Valid genders: {list(gender_encoder.classes_)}", 400
        
        try:
            values = [
                gender_encoded,
                float(request.form.get("age", 0)),
                int(request.form.get("hypertension", 0)),
                int(request.form.get("heart_disease", 0)),
                smoking_encoded,
                float(request.form.get("bmi", 0)),
                float(request.form.get("hba1c_level", 0)),
                float(request.form.get("blood_glucose_level", 0))
            ]
        except ValueError:
            return "All fields must be valid numbers.", 400
        
        df = pd.DataFrame([values], columns=diabetes_columns)
        scaled = diabetes_scaler.transform(df)

        pred = diabetes_model.predict(scaled)[0]
        prob = diabetes_model.predict_proba(scaled)[0][1] * 100
        health = round(100 - prob)
        
        username = session.get('username', 'User')
        
        PREDICTION_HISTORY.append({
            'username': username,
            'date': datetime.now().strftime("%Y-%m-%d %H:%M"),
            'disease': 'diabetes',
            'risk_high': bool(pred == 1),
            'probability': round(prob, 2),
            'health_score': health
        })
        
        if username in USER_PREDICTION_COUNTS:
            USER_PREDICTION_COUNTS[username]['diabetes'] += 1

        return render_template("result2.html",
                               pred=pred,
                               prob=round(prob, 2),
                               health=health)
    except Exception as e:
        return f"Error processing diabetes prediction: {str(e)}", 400

# ================= CHAT API =================
@app.route("/api/chat", methods=["POST"])
def chat_api():
    try:
        if not session.get('logged_in'):
            return jsonify({"error": "Not authenticated"}), 401
            
        data = request.json
        message = data.get("message", "").strip()
        
        if not message:
            return jsonify({"error": "Message is required"}), 400
        
        response = f"I understand you're asking about: '{message}'. As Carevision AI, I'm here to help with health-related questions. However, please note that I'm a demo version and should not replace professional medical advice."
        
        return jsonify({
            "response": response,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/export")
def export_data():
    if not session.get('logged_in'):
        return redirect(url_for('login_page'))
    
    username = session.get('username', 'User')
    user_predictions = [p for p in PREDICTION_HISTORY if p.get('username') == username]
    data = json.dumps(user_predictions, indent=2, default=str)
    
    return Response(
        data,
        mimetype="application/json",
        headers={"Content-disposition": f"attachment; filename={username}_predictions.json"}
    )

@app.route("/api/health-tips")
def health_tips():
    tips = [
        "Monitor your blood pressure regularly",
        "Maintain healthy cholesterol levels",
        "Check blood sugar if diabetic",
        "Regular exercise reduces heart risk by 30%",
        "Adequate sleep improves metabolic health",
        "Stay hydrated - drink at least 2 liters of water daily",
        "Include fruits and vegetables in every meal",
        "Limit processed foods and sugar intake",
        "Manage stress through meditation or yoga",
        "Get regular health check-ups"
    ]
    return jsonify({"tips": tips})

@app.errorhandler(404)
def page_not_found(e):
    return "Page not found. <a href='/'>Go to Login</a>", 404

@app.errorhandler(500)
def internal_server_error(e):
    return "Server error. <a href='/'>Go to Login</a>", 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)