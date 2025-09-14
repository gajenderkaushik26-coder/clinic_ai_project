# app.py
"""
Healthcare Clinic AI Agent (Streamlit)
- Patient chatbot (FAQs + intake form)
- AI-like triage scoring & department recommendation
- Appointment booking (SQLite persistence)
- Doctor-facing visit summaries (rule-based, optional LLM if key provided)
- Admin analytics: predicts next 7-day patient load (sklearn), shows charts

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

Deploy (Streamlit Cloud):
    1) Push repo with app.py and requirements.txt to GitHub.
    2) Create new app on share.streamlit.io from the repo.

Author: Your Team (IIM Bodh Gaya – IT Group Project)
"""

import os
import sqlite3
from datetime import datetime, date, timedelta
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# ------------------------------
# Constants & Helpers
# ------------------------------
DB_PATH = "appointments.db"

DEPARTMENTS = {
    "general": "General Medicine",
    "pediatrics": "Pediatrics",
    "ent": "ENT",
    "gyne": "Gynecology",
    "ortho": "Orthopedics",
}

FAQS = {
    "What are OPD timings?": "OPD is open from 9:00 AM to 5:00 PM, Monday to Saturday.",
    "What is the consultation fee?": "₹400 for General Medicine; specialty departments may vary.",
    "Can I reschedule an appointment?": "Yes, from the Appointments tab with your booking ID.",
    "Do you have emergency services?": "Yes, 24x7 Emergency is available at the main building.",
}

# ------------------------------
# Database
# ------------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS appointments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            age INTEGER,
            phone TEXT,
            symptoms TEXT,
            fever REAL,
            pain INTEGER,
            breathlessness INTEGER,
            chronic INTEGER,
            pregnant INTEGER,
            triage_level TEXT,
            department TEXT,
            appt_date TEXT,
            appt_time TEXT,
            created_at TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def insert_appointment(record: Dict[str, Any]) -> int:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO appointments
        (name, age, phone, symptoms, fever, pain, breathlessness, chronic, pregnant, triage_level, department, appt_date, appt_time, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record["name"],
            record["age"],
            record["phone"],
            record["symptoms"],
            record["fever"],
            record["pain"],
            int(record["breathlessness"]),
            int(record["chronic"]),
            int(record.get("pregnant", False)),
            record["triage_level"],
            record["department"],
            record["appt_date"],
            record["appt_time"],
            datetime.utcnow().isoformat(),
        ),
    )
    conn.commit()
    appt_id = cur.lastrowid
    conn.close()
    return appt_id


def load_appointments_df() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM appointments ORDER BY appt_date, appt_time", conn)
    conn.close()
    return df

# ------------------------------
# AI-ish logic (local, no external API required)
# ------------------------------
def score_triage(age: int, fever: float, pain: int, breathlessness: bool, chronic: bool, pregnant: bool) -> Dict[str, Any]:
    """Simple rule-based triage scoring.
    Returns: {level: Low/Moderate/High, dept: key, points: int}
    """
    points = 0
    # Fever
    if fever >= 38.5:
        points += 2
    elif 37.5 <= fever < 38.5:
        points += 1

    # Pain (0-10)
    if pain >= 7:
        points += 2
    elif pain >= 4:
        points += 1

    # Breathlessness
    if breathlessness:
        points += 2

    # Chronic illness
    if chronic:
        points += 1

    # Pregnancy (ages 12-55 typical range)
    if pregnant and 12 <= age <= 55:
        points += 1

    # Department heuristic
    dept = "general"
    if age <= 12:
        dept = "pediatrics"
    elif any(k in st.session_state.get("symptoms_text", "").lower() for k in ["ear", "nose", "throat", "sinus", "hearing"]):
        dept = "ent"
    elif any(k in st.session_state.get("symptoms_text", "").lower() for k in ["knee", "joint", "bone", "fracture", "sprain", "back pain"]):
        dept = "ortho"
    elif pregnant:
        dept = "gyne"

    if points >= 4:
        level = "High"
    elif points >= 2:
        level = "Moderate"
    else:
        level = "Low"

    return {"level": level, "dept": dept, "points": points}


def summarize_history(name: str, age: int, symptoms: str, chronic: bool, meds: str = "") -> str:
    # Rule-based quick summary
    lines = [
        f"Patient: {name}, Age: {age}",
        f"Chief complaints: {symptoms or 'N/A'}",
        f"History: {'Chronic condition present' if chronic else 'No known chronic illness reported'}",
    ]
    if meds:
        lines.append(f"Current meds: {meds}")
    lines.append("Triage summary auto-generated for quick review. Not a diagnosis.")
    return "\n".join(lines)


# ------------------------------
# Predictive Analytics (Admin)
# ------------------------------
def simulate_daily_counts(start_days: int = 180, seed: int = 7) -> pd.DataFrame:
    np.random.seed(seed)
    base_date = date.today() - timedelta(days=start_days)
    days = [base_date + timedelta(days=i) for i in range(start_days)]

    # seasonality: busier Mon/Tue, lighter Sun; light monthly wave
    dow = np.array([d.weekday() for d in days])  # 0=Mon
    dow_effect = np.array([1.2 if d in [0,1] else 0.9 if d in [5] else 0.8 if d in [6] else 1.0 for d in dow])
    t = np.arange(start_days)
    seasonal = 1 + 0.1 * np.sin(2 * np.pi * t / 30)

    noise = np.random.normal(0, 2.5, size=start_days)
    counts = np.clip(20 * dow_effect * seasonal + noise + 5, 5, None)
    counts = counts.astype(int)

    df = pd.DataFrame({"date": days, "count": counts})
    return df


def fit_and_forecast(df: pd.DataFrame, horizon: int = 7) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Features: day-of-week one-hot + time trend
    X_list = []
    y = df["count"].values
    for d in df["date"].dt.date:
        dow = datetime.combine(d, datetime.min.time()).weekday()
        one_hot = [1 if dow == k else 0 for k in range(7)]
        X_list.append(one_hot)
    X = np.array(X_list)

    # Add trend feature
    trend = np.arange(len(df)).reshape(-1, 1)
    X = np.hstack([X, trend])

    model = LinearRegression()
    model.fit(X, y)

    # Forecast next horizon days
    last_date = df["date"].max().date()
    fut_dates = [last_date + timedelta(days=i+1) for i in range(horizon)]
    Xf_list = []
    start_idx = len(df)
    for i, d in enumerate(fut_dates):
        dow = datetime.combine(d, datetime.min.time()).weekday()
        one_hot = [1 if dow == k else 0 for k in range(7)]
        Xf_list.append(one_hot)
    Xf = np.array(Xf_list)
    trend_f = np.arange(start_idx, start_idx + horizon).reshape(-1, 1)
    Xf = np.hstack([Xf, trend_f])

    preds = model.predict(Xf)
    preds = np.clip(np.round(preds), 0, None).astype(int)

    hist = df.copy()
    hist["date"] = pd.to_datetime(hist["date"]).dt.date

    fut = pd.DataFrame({"date": fut_dates, "forecast": preds})
    return hist, fut


# ------------------------------
# UI Components
# ------------------------------
st.set_page_config(page_title="Clinic AI Agent", page_icon="🩺", layout="wide")
init_db()

st.title("🩺 Healthcare Clinic AI Agent")
st.caption("Appointments • Triage • Summaries • Forecasts (Demo app for IT group project)")

# Sidebar Navigation
mode = st.sidebar.radio(
    "Go to",
    ["Patient Portal", "Doctor Dashboard", "Admin Analytics", "Appointments"],
)

st.sidebar.markdown("### ℹ️ FAQs")
for q, a in FAQS.items():
    with st.sidebar.expander(q):
        st.write(a)

# ------------------------------
# Patient Portal
# ------------------------------
if mode == "Patient Portal":
    st.subheader("Patient Intake & Appointment Booking")
    with st.form("intake_form"):
        cols = st.columns(3)
        name = cols[0].text_input("Full name")
        age = cols[1].number_input("Age", min_value=0, max_value=120, value=30)
        phone = cols[2].text_input("Phone")

        symptoms = st.text_area("Describe your symptoms")
        st.session_state["symptoms_text"] = symptoms

        c1, c2, c3 = st.columns(3)
        fever = c1.number_input("Fever (°C)", min_value=30.0, max_value=45.0, value=36.8, step=0.1)
        pain = c2.slider("Pain (0-10)", 0, 10, 3)
        breathlessness = c3.checkbox("Breathlessness")

        c4, c5, c6 = st.columns(3)
        chronic = c4.checkbox("Chronic illness (e.g., diabetes, asthma)")
        pregnant = c5.checkbox("Pregnant (if applicable)")
        meds = c6.text_input("Current medication (optional)")

        triage = score_triage(int(age), float(fever), int(pain), breathlessness, chronic, pregnant)
        st.info(f"Triage level: **{triage['level']}** | Suggested department: **{DEPARTMENTS[triage['dept']]}** | Points: {triage['points']}")

        # Appointment inputs
        appt_date = st.date_input("Preferred date", min_value=date.today(), value=date.today())
        appt_time = st.selectbox(
            "Preferred time",
            [
                "09:00", "09:30", "10:00", "10:30", "11:00", "11:30",
                "12:00", "12:30", "14:00", "14:30", "15:00", "15:30",
                "16:00", "16:30",
            ],
        )

        submitted = st.form_submit_button("Book Appointment")

    if submitted:
        if not name or not phone:
            st.error("Please provide your name and phone number.")
        else:
            rec = {
                "name": name,
                "age": int(age),
                "phone": phone,
                "symptoms": symptoms,
                "fever": float(fever),
                "pain": int(pain),
                "breathlessness": breathlessness,
                "chronic": chronic,
                "pregnant": pregnant,
                "triage_level": triage["level"],
                "department": triage["dept"],
                "appt_date": appt_date.isoformat(),
                "appt_time": appt_time,
            }
            appt_id = insert_appointment(rec)
            st.success(f"✅ Appointment booked! Your Booking ID is **{appt_id}**.")
            st.write("**Visit Summary (for doctor):**")
            st.code(summarize_history(name, int(age), symptoms, chronic, meds))

# ------------------------------
# Doctor Dashboard
# ------------------------------
elif mode == "Doctor Dashboard":
    st.subheader("Doctor Dashboard – Patient Summaries")
    df = load_appointments_df()
    if df.empty:
        st.info("No appointments yet.")
    else:
        # Filters
        c1, c2 = st.columns(2)
        sel_date = c1.date_input("Filter by date", value=date.today())
        dept_keys = ["All"] + list(DEPARTMENTS.keys())
        sel_dept = c2.selectbox("Department", dept_keys, index=0, format_func=lambda k: "All" if k=="All" else DEPARTMENTS[k])

        dff = df[df["appt_date"] == sel_date.isoformat()].copy()
        if sel_dept != "All":
            dff = dff[dff["department"] == sel_dept]

        if dff.empty:
            st.info("No appointments for the selected filters.")
        else:
            for _, row in dff.iterrows():
                with st.expander(f"#{row['id']} • {row['name']} • {DEPARTMENTS[row['department']]} • {row['appt_time']}"):
                    st.write(f"**Age:** {row['age']} | **Phone:** {row['phone']}")
                    st.write(f"**Triage:** {row['triage_level']} ({row['department']})")
                    st.write(f"**Symptoms:** {row['symptoms']}")
                    st.write("**Auto Summary:**")
                    st.code(summarize_history(row['name'], int(row['age']), row['symptoms'], bool(row['chronic'])))

# ------------------------------
# Admin Analytics
# ------------------------------
elif mode == "Admin Analytics":
    st.subheader("Admin Analytics – Patient Load Forecast")

    # Build history: combine simulated counts with actual booked appts per day
    sim_df = simulate_daily_counts(180)
    appt_df = load_appointments_df()

    if not appt_df.empty:
        appt_counts = appt_df.groupby("appt_date").size().rename("booked").reset_index()
        appt_counts["date"] = pd.to_datetime(appt_counts["appt_date"]).dt.date
        appt_counts = appt_counts[["date", "booked"]]
        joined = sim_df.copy()
        joined["date"] = pd.to_datetime(joined["date"]).dt.date
        joined = joined.merge(appt_counts, on="date", how="left").fillna({"booked": 0})
        joined["count"] = (joined["count"] + joined["booked"]).astype(int)
        hist, fut = fit_and_forecast(joined[["date", "count"]])
    else:
        hist, fut = fit_and_forecast(sim_df)

    # Plot history
    st.write("**Last 60 days (historical):**")
    last60 = hist.tail(60)
    fig1, ax1 = plt.subplots()
    ax1.plot(last60["date"], last60["count"], marker="o")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Patients per day")
    ax1.set_title("Historical Patient Counts (Last 60 days)")
    plt.xticks(rotation=45)
    st.pyplot(fig1)

    # Forecast chart
    st.write("**Next 7-day forecast:**")
    fig2, ax2 = plt.subplots()
    ax2.plot(fut["date"], fut["forecast"], marker="o")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Forecasted patients")
    ax2.set_title("7-Day Patient Load Forecast")
    plt.xticks(rotation=45)
    st.pyplot(fig2)

    st.dataframe(fut)

# ------------------------------
# Appointments (CRUD-lite)
# ------------------------------
else:  # "Appointments"
    st.subheader("All Appointments (Admin View)")
    df = load_appointments_df()
    if df.empty:
        st.info("No appointments stored yet.")
    else:
        st.dataframe(df)
        st.caption("Tip: Use the filters in other tabs to focus on a date/department.")

st.divider()
st.markdown(
    """
**Demo Links (to put on your PPT 'Links' slide):**
- Patient Portal (this same app) → Lets patients book appointments and get triage level.
- Doctor Dashboard → Shows auto-summaries for upcoming appointments.
- Admin Analytics → Shows historical counts and 7‑day forecast.

Once you deploy on Streamlit Cloud, copy the app URL here.
    """
)
