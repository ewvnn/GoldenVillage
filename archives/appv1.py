import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Golden Village Analytics", layout="wide")

# --- DATA LOADING (CACHED) ---
@st.cache_data
def load_data():
    """Loads all CSV data from the data/ folder. Update paths if necessary."""
    try:
        movies_df = pd.read_csv("data/gv_movies.csv")
        customers_df = pd.read_csv("data/gv_customers_v3.csv")
        segments_df = pd.read_csv("data/gv_customer_segments.csv")
        bookings_df = pd.read_csv("data/gv_bookings.csv")
        
        # Merge customers with their segments
        cust_seg_df = pd.merge(customers_df, segments_df, on="customer_id", how="left")
        
        # Merge bookings with movies for occupancy analytics
        occ_df = pd.merge(bookings_df, movies_df, on="movie_id", how="left")
        
        return movies_df, cust_seg_df, occ_df
    except FileNotFoundError as e:
        st.error(f"Data file not found: {e}. Ensure you are running this from the root directory.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

# Load data
movies_df, cust_seg_df, occ_df = load_data()

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("GV Analytics Menu")
page = st.sidebar.radio("Select Dashboard:", ["Customer Clustering", "Occupancy Analytics"])

# --- PAGE 1: CUSTOMER CLUSTERING ---
if page == "Customer Clustering":
    st.title("👥 Customer Clustering & Segmentation")
    
    if not cust_seg_df.empty:
        # --- FILTERS ---
        st.sidebar.subheader("Clustering Filters")
        
        # Safely locate column names to prevent KeyErrors
        cols = cust_seg_df.columns.tolist()
        segment_col = next((c for c in ['segment', 'cluster', 'Segment', 'Cluster', 'segment_name'] if c in cols), cols[-1])
        spend_col = next((c for c in ['total_spend', 'spend', 'Total_Spend'] if c in cols), cols[2] if len(cols) > 2 else cols[0])
        freq_col = next((c for c in ['visit_frequency', 'frequency', 'Visits'] if c in cols), cols[3] if len(cols) > 3 else cols[0])
        
        selected_segments = st.sidebar.multiselect(
            "Select Customer Segments:", 
            options=cust_seg_df[segment_col].dropna().unique(),
            default=cust_seg_df[segment_col].dropna().unique()
        )
        
        filtered_cust = cust_seg_df[cust_seg_df[segment_col].isin(selected_segments)]
        
        # --- KPIs ---
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Customers", f"{len(filtered_cust):,}")
        col2.metric("Average Spend", f"${filtered_cust[spend_col].mean():.2f}")
        col3.metric("Avg Visits", f"{filtered_cust[freq_col].mean():.1f}")
        
        st.markdown("---")
        
        # --- VISUALIZATIONS ---
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.subheader("Segment Distribution")
            fig_bar = px.bar(filtered_cust[segment_col].value_counts().reset_index(), 
                             x=segment_col, y='count', 
                             labels={segment_col: 'Segment', 'count': 'Number of Customers'},
                             color=segment_col)
            st.plotly_chart(fig_bar, use_container_width=True)
            
        with col_chart2:
            st.subheader("Spend vs. Frequency by Segment")
            fig_scatter = px.scatter(filtered_cust, x=freq_col, y=spend_col, 
                                     color=segment_col, hover_data=['customer_id'],
                                     opacity=0.7)
            st.plotly_chart(fig_scatter, use_container_width=True)

# --- PAGE 2: OCCUPANCY ANALYTICS ---
elif page == "Occupancy Analytics":
    st.title("🍿 Occupancy Analytics")
    
    if not occ_df.empty:
        # Safely locate column names to prevent KeyErrors
        cols = occ_df.columns.tolist()
        genre_col = next((c for c in ['genre', 'movie_genre', 'Genre'] if c in cols), cols[1] if len(cols) > 1 else cols[0])
        day_col = next((c for c in ['day_of_week', 'date', 'Date'] if c in cols), cols[2] if len(cols) > 2 else cols[0])
        occ_rate_col = next((c for c in ['occupancy_rate', 'tickets_sold', 'Occupancy'] if c in cols), cols[-1]) 
        
        # --- FILTERS (Independent Variables) ---
        st.sidebar.subheader("Occupancy Filters")
        
        if genre_col in occ_df.columns:
            selected_genres = st.sidebar.multiselect(
                "Filter by Genre:", 
                options=occ_df[genre_col].dropna().unique(),
                default=occ_df[genre_col].dropna().unique()
            )
        else:
            selected_genres = []

        if day_col in occ_df.columns:
            selected_days = st.sidebar.multiselect(
                "Filter by Day:", 
                options=occ_df[day_col].dropna().unique(),
                default=occ_df[day_col].dropna().unique()
            )
        else:
            selected_days = []

        # Apply filters
        filtered_occ = occ_df.copy()
        if selected_genres:
            filtered_occ = filtered_occ[filtered_occ[genre_col].isin(selected_genres)]
        if selected_days:
            filtered_occ = filtered_occ[filtered_occ[day_col].isin(selected_days)]
        
        # --- KPIs ---
        col1, col2, col3 = st.columns(3)
        total_bookings = len(filtered_occ)
        avg_occupancy = filtered_occ[occ_rate_col].mean() if occ_rate_col in filtered_occ.columns else 0
        top_movie = filtered_occ['title'].value_counts().index[0] if 'title' in filtered_occ.columns else "N/A"
        
        col1.metric("Total Bookings", f"{total_bookings:,}")
        col2.metric("Avg. Occupancy Rate", f"{avg_occupancy:.1f}%" if avg_occupancy > 0 else "N/A")
        col3.metric("Top Movie", top_movie)
        
        st.markdown("---")
        
        # --- VISUALIZATIONS ---
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.subheader("Occupancy by Genre")
            if genre_col in filtered_occ.columns and occ_rate_col in filtered_occ.columns:
                genre_grp = filtered_occ.groupby(genre_col)[occ_rate_col].mean().reset_index()
                fig_genre = px.bar(genre_grp, x=genre_col, y=occ_rate_col, color=genre_col)
                st.plotly_chart(fig_genre, use_container_width=True)
            else:
                st.info("Genre or Occupancy Rate columns not found for this chart.")
                
        with col_chart2:
            st.subheader("Occupancy Trends (Day of Week/Date)")
            if day_col in filtered_occ.columns and occ_rate_col in filtered_occ.columns:
                trend_grp = filtered_occ.groupby(day_col)[occ_rate_col].mean().reset_index()
                fig_trend = px.line(trend_grp, x=day_col, y=occ_rate_col, markers=True)
                st.plotly_chart(fig_trend, use_container_width=True)
            else:
                st.info("Date/Day or Occupancy Rate columns not found for this chart.")
                
        # Optional: Show raw data toggle
        if st.checkbox("Show Raw Data"):
            st.dataframe(filtered_occ)