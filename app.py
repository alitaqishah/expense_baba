import streamlit as st
import pandas as pd
import os
from groq import Groq
import matplotlib.pyplot as plt
from datetime import datetime

# ---------------------------------------
# Initialize Groq Client
# ---------------------------------------


GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found. Please set it in environment variables.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# ---------------------------------------
# Load or Create Dataset
# ---------------------------------------
DATA_FILE = "expenses.csv"

if os.path.exists(DATA_FILE):
    df = pd.read_csv(DATA_FILE)
else:
    df = pd.DataFrame(columns=["date", "amount", "description", "category"])

# ---------------------------------------
# AI Category Prediction
# ---------------------------------------
def get_ai_category(description):
    prompt = f"""
    Categorize this expense into one of the following:
    Food, Transport, Shopping, Bills, Entertainment, Health, Education,
    Subscriptions, Miscellaneous.
    Description: "{description}"
    Respond ONLY with the category name.
    """

    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile"
    )
    return response.choices[0].message.content.strip()

# ---------------------------------------
# AI Expense Analysis
# ---------------------------------------
def analyze_expenses(df):
    if df.empty:
        return "No expenses added yet."

    summary = df.groupby("category")["amount"].sum().sort_values(ascending=False)
    total = df["amount"].sum()

    prompt = f"""
    Analyze the following spending data.
    Total Spending: {total}
    Breakdown by Category: {summary.to_dict()}
    Provide:
    - Spending behavior
    - Overspending categories
    - Tips to save money
    - Budgeting strategies
    Write in bullet points.
    """

    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile"
    )

    return response.choices[0].message.content

# ---------------------------------------
# Streamlit UI
# ---------------------------------------
st.title("💰 Expense Analyzer Baba")

st.subheader("➕ Add Expense")

with st.form("expense_form"):
    amount = st.number_input("Amount", min_value=1.0, format="%.2f")
    description = st.text_input("Description")
    date = st.date_input("Date", value=datetime.today())
    submitted = st.form_submit_button("Add Expense")

if submitted:
    ai_category = get_ai_category(description)

    new_row = pd.DataFrame(
        [[str(date), amount, description, ai_category]],
        columns=["date", "amount", "description", "category"]
    )

    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(DATA_FILE, index=False)

    st.success(f"Expense added under category: **{ai_category}**")

st.divider()

# ---------------------------------------
# Expense Table
# ---------------------------------------
st.subheader("📘 Expense Records")
st.dataframe(df, use_container_width=True)

# ---------------------------------------
# Charts
# ---------------------------------------
if not df.empty:

    st.subheader("📊 Expense Visualizations")

    col1, col2 = st.columns(2)

    # Pie Chart
    with col1:
        st.write("### Category Distribution")
        fig1, ax1 = plt.subplots()
        df.groupby("category")["amount"].sum().plot.pie(
            autopct="%1.1f%%", ax=ax1
        )
        ax1.set_ylabel("")
        st.pyplot(fig1)

    # Bar Chart
    with col2:
        st.write("### Spending By Category")
        fig2, ax2 = plt.subplots()
        df.groupby("category")["amount"].sum().plot.bar(ax=ax2)
        st.pyplot(fig2)

    # Trend Line
    st.write("### Spending Trend Over Time")
    fig3, ax3 = plt.subplots()
    df["date"] = pd.to_datetime(df["date"])
    df.groupby("date")["amount"].sum().plot(ax=ax3)
    ax3.set_ylabel("Amount Spent")
    st.pyplot(fig3)

# ---------------------------------------
# AI Insights
# ---------------------------------------
st.divider()
st.subheader("🤖 AI Insights & Recommendations")

if st.button("Generate AI Insights"):
    insights = analyze_expenses(df)
    st.write(insights)

