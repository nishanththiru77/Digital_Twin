import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# Import sub-agents
from water_agent import predict_water_storage, train_water_model, MODEL_PATH
from electricity_agent import electricity_agent
from crop_agent import crop_agent

# Load environment variables
load_dotenv()

# Streamlit Page Config
st.set_page_config(page_title="AI Village Digital Twin", page_icon="🏘️", layout="wide")

# Custom Styles
st.markdown("""
<style>
    .main { background-color: #f0f2f6; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 12px; box-shadow: 0 4px 8px rgba(0,0,0,0.05); }
    .recommendation-box { background-color: #e3f2fd; padding: 25px; border-radius: 15px; border-left: 8px solid #1976d2; }
    .eval-box { background-color: #f1f8e9; padding: 15px; border-radius: 10px; border: 1px solid #c5e1a5; margin-top: 10px; }
</style>
""", unsafe_allow_html=True)

def main():
    st.title("🏘️ Village Management Digital Twin")
    st.markdown("### AI-Powered Strategic Decision Support System")

    # Sidebar Inputs
    with st.sidebar:
        st.header("📍 Village Parameters")

        st.markdown("### 💧 Water Parameters")
        population = st.number_input("Village Population", min_value=1, value=1500)
        roof_area = st.number_input("Average Roof Area (sqm)", min_value=1.0, value=120.0)
        rainfall = st.number_input("Rainfall (mm)", min_value=0.0, value=45.0)
        tank_capacity = st.number_input("Storage Tank Capacity (Liters)", min_value=100.0, value=2000.0)

        st.markdown("### 🌦 Weather Parameters")
        temperature = st.number_input("Temperature (°C)", min_value=-5.0, max_value=55.0, value=32.0)
        humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0)

        st.markdown("### ⚡ Electricity Parameters")
        pump_usage = st.selectbox("Water Pump Usage", ["Low", "Medium", "High"])
        month_name = st.selectbox("Month", ["January","February","March","April","May","June","July","August","September","October","November","December"])
        month_map = {"January":1,"February":2,"March":3,"April":4,"May":5,"June":6,"July":7,"August":8,"September":9,"October":10,"November":11,"December":12}
        month = month_map[month_name]

        st.markdown("### 🌾 Agriculture Parameters")
        soil_moisture = st.number_input("Soil Moisture (%)", min_value=0.0, max_value=100.0, value=60.0)
        fertilizer_used = st.number_input("Fertilizer Used (kg)", min_value=0.0, value=150.0)

        st.markdown("---")

        analyze_btn = st.button("🚀 Analyze Village", use_container_width=True)

        if st.button("🔄 Retrain Water Model"):
            with st.spinner("Training model..."):
                _, r2, mae = train_water_model()
                st.success(f"Model Retrained!\nR²: {r2:.4f}\nMAE: {mae:.2f}")

    if analyze_btn:
        with st.spinner("Processing Agent Data..."):

            # ===============================
            # WATER AGENT
            # ===============================
            water_results = predict_water_storage(population, roof_area, rainfall, tank_capacity)

            # ===============================
            # ELECTRICITY AGENT
            # ===============================
            electricity_results = electricity_agent(
                temperature=temperature,
                humidity=humidity,
                population=population,
                pump_usage=pump_usage,
                month=month
            )

            # ===============================
            # CROP AGENT
            # ===============================
            crop_results = crop_agent(
                rainfall=rainfall,
                temperature=temperature,
                soil_moisture=soil_moisture,
                fertilizer_used=fertilizer_used
            )

            # ===============================
            # MODEL PERFORMANCE
            # ===============================
            if os.path.exists(MODEL_PATH):
                _, r2, mae = train_water_model()
                st.markdown(
                    f"""
                    <div class="eval-box">
                    <strong>Water Model Performance:</strong> R² Score: {r2:.4f} | MAE: {mae:.2f}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # ===============================
            # ANALYSIS INSIGHTS
            # ===============================
            st.markdown("## 📊 Analysis Insights")

            # WATER ANALYSIS
            st.markdown("### 💧 Water Analysis")
            w1, w2, w3 = st.columns(3)
            with w1:
                st.metric("Predicted Water Collected", f"{water_results['predicted_water_storage']} L")
            with w2:
                st.metric("Required Water", f"{water_results['required_water']} L")
            with w3:
                st.metric("Water Deficit", f"{water_results['water_deficit']} L")

            # ELECTRICITY ANALYSIS
            st.markdown("### ⚡ Electricity Analysis")
            e1, e2, e3 = st.columns(3)
            with e1:
                st.metric("Predicted Consumption", f"{electricity_results['predicted_consumption']} kWh")
            with e2:
                st.metric("Transformer Capacity", f"{electricity_results['transformer_capacity']} kWh")
            with e3:
                st.metric("Utilization", f"{electricity_results['utilization_percent']} %")

            # CROP ANALYSIS
            st.markdown("### 🌾 Crop Analysis")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Predicted Yield", f"{crop_results['predicted_yield']} kg")
            with c2:
                st.metric("Expected Yield", f"{crop_results['expected_yield']} kg")
            with c3:
                st.metric("Yield Achievement", f"{crop_results['yield_percentage']} %")

            # ===============================
            # RISK ASSESSMENT
            # ===============================
            st.markdown("## 🚨 Risk Assessment")
            r1, r2, r3 = st.columns(3)
            with r1:
                st.metric("Water Risk", water_results["water_risk"])
            with r2:
                st.metric("Electricity Risk", electricity_results["electricity_risk"])
            with r3:
                st.metric("Crop Risk", crop_results["crop_risk"])

            # ===============================
            # GEMINI RECOMMENDATIONS
            # ===============================
            st.markdown("---")
            st.subheader("💡 Strategic Recommendations (Gemini AI)")

            recommendations = get_ai_recommendations(
                water_results["predicted_water_storage"],
                water_results["required_water"],
                water_results["water_deficit"],
                water_results["water_risk"],
                electricity_results["electricity_risk"],
                crop_results["crop_risk"]
            )

            st.markdown(
                f"""
                <div class="recommendation-box">
                {recommendations}
                </div>
                """,
                unsafe_allow_html=True
            )

def get_ai_recommendations(storage, required, deficit, w_risk, e_risk, c_risk):
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return "⚠️ Please set GOOGLE_API_KEY in your .env file to enable AI recommendations."
    
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
        
        prompt = f"""
        Act as the Chief Village Manager. We have the following Digital Twin simulation results:
        
        - Water Collected: {storage} L
        - Required Water: {required} L
        - Water Deficit: {deficit} L
        - Water Risk Level: {w_risk}
        - Electricity Risk: {e_risk}
        - Crop Risk: {c_risk}
        
        Provide 5 specific, professional, and actionable steps for the village council. 
        Focus on immediate mitigation if risks are Medium/High.
        Format as a clean bulleted list.
        """
        
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"Error connecting to Gemini: {str(e)}"

if __name__ == "__main__":
    main()
