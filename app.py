import cv2
import streamlit as st
from utils.database_handler import get_item_footprint, load_water_data
from utils.vision_engine import load_yolo_model, process_frame

# Page Configuration
st.set_page_config(
    page_title="Water Footprint Detector", page_icon="💧", layout="centered"
)

st.title("💧 Daily Object Water Footprint Calculator")
st.write(
    "Analyze the environmental impact of everyday items using manual search or"
    " live camera detection."
)

# Load Database and Model
water_db = load_water_data()


@st.cache_resource
def get_model():
  return load_yolo_model()


model = get_model()

# Navigation Mode Option
app_mode = st.radio(
    "Select Mode:", ("🔍 Manual Search Bar", "📷 Live Camera Detection")
)

# ==================== MODE 1: MANUAL SEARCH ====================
if app_mode == "🔍 Manual Search Bar":
  st.subheader("Manual Object Search")
  search_query = st.text_input(
      "Type an everyday item (e.g., bottle, cup, laptop, cotton t-shirt):"
  )

  if search_query:
    item_data = get_item_footprint(search_query)
    if item_data:
      st.success(f"Found footprint data for **{search_query.title()}**")

      # Tabs for overview and breakdown under the search bar
      tab_summary, tab_breakdown = st.tabs(
          ["📊 Footprint Overview", "🔬 Detailed Breakdown"]
      )

      with tab_summary:
        st.metric(
            label="Total Water Footprint",
            value=f"{item_data['total']} Liters",
        )

        col1, col2, col3 = st.columns(3)
        col1.metric("💙 Blue Water", f"{item_data['blue']} L")
        col2.metric("💚 Green Water", f"{item_data['green']} L")
        col3.metric("🩶 Grey Water", f"{item_data['grey']} L")

      with tab_breakdown:
        st.markdown(f"### Component Breakdown for **{search_query.title()}**:")
        st.write(
            f"- **Blue Water ({item_data['blue']} L):** Surface and groundwater"
            " consumed during manufacturing/growth."
        )
        st.write(
            f"- **Green Water ({item_data['green']} L):** Rainwater consumed"
            " (crucial for agriculture & textiles)."
        )
        st.write(
            f"- **Grey Water ({item_data['grey']} L):** Fresh water required"
            " to dilute pollutants back to standards."
        )

        # Visual Bar Chart Breakdown
        chart_data = {
            "Water Type": ["Blue Water", "Green Water", "Grey Water"],
            "Liters": [
                item_data["blue"],
                item_data["green"],
                item_data["grey"],
            ],
        }
        st.bar_chart(chart_data, x="Water Type", y="Liters")

    else:
      st.warning(
          "Item not found in database. Try searching for 'bottle', 'cup',"
          " 'laptop', or 'cotton t-shirt'."
      )

# ==================== MODE 2: LIVE CAMERA ====================
elif app_mode == "📷 Live Camera Detection":
  st.subheader("Live Webcam Detection")
  st.write(
      "Toggle the checkbox below to initialize your webcam and spot objects"
      " in real-time."
  )

  run = st.checkbox("Turn on Camera")
  FRAME_WINDOW = st.image([])
  detected_placeholder = st.empty()

  cap = cv2.VideoCapture(0)

  while run:
    ret, frame = cap.read()
    if not ret:
      st.error(
          "Failed to access webcam. Please check your camera connection or"
          " browser permissions."
      )
      break

    # Process frame through YOLO vision engine
    annotated_frame, detected_items = process_frame(frame, model, water_db)

    # Convert OpenCV BGR format to RGB for Streamlit UI rendering
    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(annotated_frame)

    # Display real-time result logs underneath the video feed
    if detected_items:
      with detected_placeholder.container():
        st.info(
            "🔍 Detected Objects: "
            + ", ".join([item.title() for item in detected_items])
        )
        for item in detected_items:
          data = water_db[item]
          st.write(
              f"• **{item.title()}** ➔ Total: **{data['total']} L**"
              f" (💙 Blue: {data['blue']}L | 💚 Green: {data['green']}L | 🩶"
              f" Grey: {data['grey']}L)"
          )
    else:
      detected_placeholder.empty()

  cap.release()