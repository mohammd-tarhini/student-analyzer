import streamlit as st
import requests

# ==========================================
# FastAPI Backend URL Configuration
# ==========================================
BASE_URL = "http://127.0.0.1:8000"

# ==========================================
# Streamlit Page Configuration
# ==========================================
st.set_page_config(
    page_title="StudyPulse - Student Analytics",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 StudyPulse - Student Analytics Dashboard")
st.markdown("A professional student performance analytics dashboard powered by Machine Learning and FastAPI.")

# ==========================================
# Sidebar: Search for Existing Student Analysis
# ==========================================
st.sidebar.header("Search Student Records")
search_id = st.sidebar.text_input("Enter Student ID:")

if st.sidebar.button("Search"):
    if search_id:
        try:
            response = requests.get(f"{BASE_URL}/student/{search_id}/analysis")
            if response.status_code == 200:
                student_data = response.json()
                st.sidebar.success("Student record found!")
                st.sidebar.json(student_data)
            else:
                st.sidebar.error("Student ID not found in the database.")
        except Exception as e:
            st.sidebar.error(f"Connection failed: {e}")
    else:
        st.sidebar.warning("Please enter a valid Student ID.")

st.markdown("---")

# ==========================================
# Main Form: Enter Student Data & Run Analysis
# ==========================================
st.header("New Student Analysis Form")

with st.form("student_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        student_id = st.text_input("Student ID", "STU-1024")
        hours_studied = st.number_input("Hours Studied", min_value=0.0, max_value=24.0, value=5.0, step=0.5)
        attendance = st.number_input("Attendance (%)", min_value=0.0, max_value=100.0, value=80.0, step=1.0)
        parental_involvement = st.selectbox("Parental Involvement", ["Low", "Medium", "High"])
    
    with col2:
        student_name = st.text_input("Student Name", "Ahmad Ali")
        sleep_hours = st.number_input("Sleep Hours", min_value=0.0, max_value=24.0, value=7.0, step=0.5)
        tutoring_sessions = st.number_input("Tutoring Sessions", min_value=0, max_value=10, value=1, step=1)
        access_to_internet = st.checkbox("Access to Internet", value=True)

    # Form submission button
    submit_button = st.form_submit_button(label="Analyze & Save Data")

# ==========================================
# Handle Form Submission & API Communication
# ==========================================
if submit_button:
    # Prepare payload for FastAPI backend
    payload = {
        "student_id": student_id,
        "student_name": student_name,
        "Hours_Studied": hours_studied,
        "Attendance": attendance,
        "Parental_Involvement": parental_involvement,
        "Access_to_Internet": access_to_internet,
        "Sleep_Hours": sleep_hours,
        "Tutoring_Sessions": tutoring_sessions
    }

    try:
        # Send POST request to FastAPI /analyze endpoint
        res = requests.post(f"{BASE_URL}/analyze", json=payload)
        
        if res.status_code == 200:
            result = res.json()
            
            st.success("Student data successfully analyzed and stored in PostgreSQL!")
            
            # Display results summary
            st.subheader(f"Analysis Results for: {result['student_name']} (ID: {result['student_id']})")
            
            # Metrics display
            m1, m2, m3 = st.columns(3)
            m1.metric("Predicted Exam Score", result["predicted_exam_score"])
            m2.metric("Performance Level", result["predicted_level"])
            m3.metric("Student Cluster", result["student_cluster"])
            
            # Recommendations section
            st.markdown("### Personalized Recommendations:")
            for rec in result["recommendations"]:
                st.info(rec)
                
        else:
            st.error(f"Server Error: {res.text}")
            
    except Exception as e:
        st.error(f"Could not connect to the FastAPI backend: {e}")