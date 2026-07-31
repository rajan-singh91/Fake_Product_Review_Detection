from flask import Flask, render_template, request, redirect, url_for, session
import pickle
import sqlite3
from datetime import datetime
import matplotlib.pyplot as plt
import os

app = Flask(__name__)
app.secret_key = "fake_review_project_2026"

# -----------------------------
# Load ML Model
# -----------------------------

model = pickle.load(
    open("model/fake_review_model.pkl", "rb")
)

vectorizer = pickle.load(
    open("model/vectorizer.pkl", "rb")
)

# -----------------------------
# Create Database
# -----------------------------

def create_database():

    if not os.path.exists("database"):
        os.makedirs("database")

    conn = sqlite3.connect("database/reviews.db")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS history(

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            review TEXT,

            result TEXT,

            confidence REAL,

            date TEXT

        )
    """)

    conn.commit()
    conn.close()

create_database()


# -----------------------------
# Pie Chart
# -----------------------------

def create_chart(real, fake):

    plt.figure(figsize=(5,5))

    plt.pie(
        [real, fake],
        labels=["Real","Fake"],
        autopct="%1.1f%%"
    )

    plt.title("Review Analysis")

    plt.savefig("static/chart.png")

    plt.close()


@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        username = request.form["username"]

        password = request.form["password"]

        if username == "admin" and password == "admin123":

            session["user"] = username

            return redirect(url_for("home"))

        else:

            return render_template(
                "login.html",
                error="Invalid Username or Password"
            )

    return render_template("login.html")

# -----------------------------
# Home Page
# -----------------------------

@app.route("/")
def home():

    if "user" not in session:
        return redirect(url_for("login"))

    conn = sqlite3.connect("database/reviews.db")

    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM history")
    total = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM history WHERE result='real'")
    real = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM history WHERE result='fake'")
    fake = cursor.fetchone()[0]

    conn.close()

    if total > 0:
        create_chart(real, fake)

    return render_template(

        "index.html",

        total=total,

        real=real,

        fake=fake

    )

# -----------------------------
# Prediction
# -----------------------------

@app.route("/predict", methods=["POST"])
def predict():

    review = request.form["review"]

    review_vector = vectorizer.transform([review])

    prediction = model.predict(review_vector)

    result = prediction[0]

    probability = model.predict_proba(review_vector)

    confidence = round(max(probability[0]) * 100, 2)

    # Save Review

    conn = sqlite3.connect("database/reviews.db")

    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO history
        (review, result, confidence, date)
        VALUES (?, ?, ?, ?)
        """,
        (
            review,
            result,
            confidence,
            datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        )
    )

    conn.commit()

    # Dashboard Count

    cursor.execute("SELECT COUNT(*) FROM history")
    total = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM history WHERE result='real'")
    real = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM history WHERE result='fake'")
    fake = cursor.fetchone()[0]

    conn.close()

    create_chart(real, fake)

    return render_template(
        "index.html",
        prediction=result,
        confidence=confidence,
        review=review,
        total=total,
        real=real,
        fake=fake
    )


# -----------------------------
# History Page
# -----------------------------

@app.route("/history")
def history():

    conn = sqlite3.connect("database/reviews.db")

    cursor = conn.cursor()

    cursor.execute(
        "SELECT * FROM history ORDER BY id DESC"
    )

    history = cursor.fetchall()

    conn.close()

    return render_template(
        "history.html",
        history=history
    )

# -----------------------------
# Delete History
# -----------------------------

@app.route("/clear")
def clear():

    conn = sqlite3.connect("database/reviews.db")

    cursor = conn.cursor()

    cursor.execute("DELETE FROM history")

    conn.commit()

    conn.close()

    # Delete old chart if exists
    chart_path = "static/chart.png"

    if os.path.exists(chart_path):
        os.remove(chart_path)

    return home()

@app.route("/logout")
def logout():

    session.clear()

    return redirect(url_for("login"))


# -----------------------------
# About Page
# -----------------------------

@app.route("/about")
def about():

    return """
    <h2>Fake Product Review Detection</h2>

    <p>
    This project is developed using
    Flask, Machine Learning,
    SQLite and TF-IDF.
    </p>

    <a href="/">Go Back</a>
    """


# -----------------------------
# Run Flask App
# -----------------------------

if __name__ == "__main__":

    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000
    )