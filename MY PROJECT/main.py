from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import psycopg2


# ==========================================
# PostgreSQL Database Connection
# ==========================================

def get_db_connection():
    connection = psycopg2.connect(
        host="localhost",
        database="your database name",
        user="postgres",
        password="your password",
        port="5432"
    )
    return connection


# ==========================================
# FastAPI App Initialization
# ==========================================

app = FastAPI()


# ==========================================
# Load Machine Learning Models
# ==========================================

reg_model = joblib.load("regression_model.pkl")
clf_model = joblib.load("classification_model.pkl")
kmeans_model = joblib.load("kmeans_model.pkl")


# ==========================================
# Load Encoders
# ==========================================

parental_encoder = joblib.load("parental_encoder.pkl")
internet_encoder = joblib.load("internet_encoder.pkl")


# ==========================================
# Student Input Schema (Pydantic Model)
# ==========================================

class student(BaseModel):
    student_id: str
    student_name: str
    Hours_Studied: float
    Attendance: float
    Parental_Involvement: str
    Access_to_Internet: bool
    Sleep_Hours: float
    Tutoring_Sessions: int


# ==========================================
# Predict Exam Score & Save to Database
# ==========================================

@app.post("/predict/score")
def predict_score(data: student):

    features = [[
        data.Hours_Studied,
        data.Attendance,
        data.Sleep_Hours,
        data.Tutoring_Sessions
    ]]

    score = reg_model.predict(features)
    predicted_score = round(float(score[0]), 2)

    # Save data and prediction to PostgreSQL
    try:
        connection = get_db_connection()
        cursor = connection.cursor()
        
        insert_query = """
            INSERT INTO student_analysis (
                student_id, student_name, hours_studied, attendance,
                parental_involvement, access_to_internet, sleep_hours,
                tutoring_sessions, predicted_score
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        cursor.execute(insert_query, (
            data.student_id, 
            data.student_name, 
            data.Hours_Studied,
            data.Attendance, 
            data.Parental_Involvement, 
            data.Access_to_Internet,
            data.Sleep_Hours, 
            data.Tutoring_Sessions, 
            predicted_score
        ))
        
        connection.commit()
        cursor.close()
        connection.close()
        
    except Exception as e:
        print(f"Database Error: {e}")

    return {
        "student_id": data.student_id,
        "student_name": data.student_name,
        "predicted_exam_score": predicted_score
    }


# ==========================================
# Predict Performance Level & Save to Database
# ==========================================

@app.post("/predict/levels")
def predicted_levels(data: student):

    # Encode Parental Involvement
    parental_encoded = parental_encoder.transform(
        [data.Parental_Involvement]
    )[0]

    # Convert Boolean to Integer for model compatibility
    internet_encoded = int(data.Access_to_Internet)

    features = [[
        data.Hours_Studied,
        data.Attendance,
        parental_encoded,
        internet_encoded,
        data.Sleep_Hours,
        data.Tutoring_Sessions
    ]]

    level = clf_model.predict(features)
    predicted_level = int(level[0])

    # Save data and prediction to PostgreSQL
    try:
        connection = get_db_connection()
        cursor = connection.cursor()
        
        insert_query = """
            INSERT INTO student_analysis (
                student_id, student_name, hours_studied, attendance,
                parental_involvement, access_to_internet, sleep_hours,
                tutoring_sessions, predicted_level
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        cursor.execute(insert_query, (
            data.student_id, 
            data.student_name, 
            data.Hours_Studied,
            data.Attendance, 
            data.Parental_Involvement, 
            data.Access_to_Internet,
            data.Sleep_Hours, 
            data.Tutoring_Sessions, 
            predicted_level
        ))
        
        connection.commit()
        cursor.close()
        connection.close()
        
    except Exception as e:
        print(f"Database Error: {e}")

    return {
        "student_id": data.student_id,
        "student_name": data.student_name,
        "predicted_level": predicted_level
    }


# ==========================================
# Predict Student Cluster & Save to Database
# ==========================================

@app.post("/predict/cluster")
def predict_cluster(data: student):

    reg_features = [[
        data.Hours_Studied,
        data.Attendance,
        data.Sleep_Hours,
        data.Tutoring_Sessions
    ]]

    predicted_score = reg_model.predict(reg_features)[0]

    cluster_features = [[
        data.Hours_Studied,
        predicted_score
    ]]

    cluster_id = kmeans_model.predict(cluster_features)
    student_cluster = int(cluster_id[0])

    # Save data and prediction to PostgreSQL
    try:
        connection = get_db_connection()
        cursor = connection.cursor()
        
        insert_query = """
            INSERT INTO student_analysis (
                student_id, student_name, hours_studied, attendance,
                parental_involvement, access_to_internet, sleep_hours,
                tutoring_sessions, student_cluster
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        cursor.execute(insert_query, (
            data.student_id, 
            data.student_name, 
            data.Hours_Studied,
            data.Attendance, 
            data.Parental_Involvement, 
            data.Access_to_Internet,
            data.Sleep_Hours, 
            data.Tutoring_Sessions, 
            student_cluster
        ))
        
        connection.commit()
        cursor.close()
        connection.close()
        
    except Exception as e:
        print(f"Database Error: {e}")

    return {
        "student_id": data.student_id,
        "student_name": data.student_name,
        "student_cluster": student_cluster
    }


# ==========================================
# Comprehensive Student Analysis & Save to DB
# ==========================================

@app.post("/analyze")
def analyze_student(data: student):

    # --------------------------------------
    # 1. Predict Exam Score
    # --------------------------------------

    reg_features = [[
        data.Hours_Studied,
        data.Attendance,
        data.Sleep_Hours,
        data.Tutoring_Sessions
    ]]

    predicted_score = round(float(reg_model.predict(reg_features)[0]), 2)


    # --------------------------------------
    # 2. Predict Performance Level
    # --------------------------------------

    parental_encoded = parental_encoder.transform(
        [data.Parental_Involvement]
    )[0]
    
    internet_encoded = int(data.Access_to_Internet)

    clf_features = [[
        data.Hours_Studied,
        data.Attendance,
        parental_encoded,
        internet_encoded,
        data.Sleep_Hours,
        data.Tutoring_Sessions
    ]]

    predicted_level = int(clf_model.predict(clf_features)[0])


    # --------------------------------------
    # 3. Predict Student Cluster
    # --------------------------------------

    cluster_features = [[
        data.Hours_Studied,
        predicted_score
    ]]

    student_cluster = int(kmeans_model.predict(cluster_features)[0])


    # --------------------------------------
    # 4. Generate Recommendations
    # --------------------------------------

    recommendations = []

    if data.Hours_Studied < 5:
        recommendations.append(
            "Increase daily study hours to improve overall performance."
        )

    if data.Attendance < 75:
        recommendations.append(
            "Improve class attendance to better understand core concepts."
        )

    if data.Tutoring_Sessions == 0 and predicted_score < 60:
        recommendations.append(
            "Consider joining tutoring sessions for academic support."
        )

    if data.Sleep_Hours < 6:
        recommendations.append(
            "Ensure adequate sleep (at least 7-8 hours) for optimal cognitive function."
        )

    if not recommendations:
        recommendations.append(
            "Great job! Maintain your current study routine and habits."
        )


    # --------------------------------------
    # 5. Save Complete Analysis to PostgreSQL
    # --------------------------------------

    try:
        connection = get_db_connection()
        cursor = connection.cursor()

        insert_query = """
            INSERT INTO student_analysis (
                student_id, student_name, hours_studied, attendance,
                parental_involvement, access_to_internet, sleep_hours,
                tutoring_sessions, predicted_score, predicted_level, student_cluster
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        cursor.execute(insert_query, (
            data.student_id,
            data.student_name,
            data.Hours_Studied,
            data.Attendance,
            data.Parental_Involvement,
            data.Access_to_Internet,
            data.Sleep_Hours,
            data.Tutoring_Sessions,
            predicted_score,
            predicted_level,
            student_cluster
        ))

        connection.commit()
        cursor.close()
        connection.close()

    except Exception as e:
        print(f"Database Error: {e}")


    # --------------------------------------
    # 6. Return Complete Analysis Response
    # --------------------------------------

    return {
        "student_id": data.student_id,
        "student_name": data.student_name,
        "predicted_exam_score": predicted_score,
        "predicted_level": predicted_level,
        "student_cluster": student_cluster,
        "recommendations": recommendations
    }


# ==========================================
# Retrieve Student Analysis by Student ID
# ==========================================

@app.get("/student/{student_id}/analysis")
def get_student_analysis(student_id: str):
    
    try:
        connection = get_db_connection()
        cursor = connection.cursor()
        
        query = """
            SELECT student_id, student_name, hours_studied, attendance,
                   parental_involvement, access_to_internet, sleep_hours,
                   tutoring_sessions, predicted_score, predicted_level, 
                   student_cluster, created_at
            FROM student_analysis
            WHERE student_id = %s
            ORDER BY created_at DESC
            LIMIT 1;
        """
        
        cursor.execute(query, (student_id,))
        row = cursor.fetchone()
        
        cursor.close()
        connection.close()
        
        if not row:
            return {"error": "Student record not found"}
            
        return {
            "student_id": row[0],
            "student_name": row[1],
            "hours_studied": row[2],
            "attendance": row[3],
            "parental_involvement": row[4],
            "access_to_internet": row[5],
            "sleep_hours": row[6],
            "tutoring_sessions": row[7],
            "predicted_score": row[8],
            "predicted_level": row[9],
            "student_cluster": row[10],
            "created_at": str(row[11])
        }
        
    except Exception as e:
        return {"error": str(e)}