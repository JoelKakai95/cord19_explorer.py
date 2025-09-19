import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from datetime import datetime
import re
from collections import Counter

# Set page configuration
st.set_page_config(
    page_title="CORD-19 Data Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {font-size: 3rem; color: #1f77b4; margin-bottom: 1rem;}
    .section-header {font-size: 2rem; color: #1f77b4; border-bottom: 2px solid #1f77b4; padding-bottom: 0.5rem;}
    .info-text {background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;}
    </style>
    """, unsafe_allow_html=True)

# Title and introduction
st.markdown('<h1 class="main-header">CORD-19 Data Explorer</h1>', unsafe_allow_html=True)
st.markdown("""
    <div class="info-text">
    This application provides an interactive exploration of the CORD-19 dataset, which contains metadata 
    about COVID-19 research papers. Use the sidebar filters to explore the data and visualize trends in 
    COVID-19 research publications.
    </div>
    """, unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Filters and Controls")
st.sidebar.markdown("Use these controls to filter and explore the dataset.")

# Simulate loading data (in a real scenario, we would load from metadata.csv)
@st.cache_data
def load_data():
    # Create sample data for demonstration
    # In a real implementation, we would use: df = pd.read_csv('metadata.csv')
    np.random.seed(42)
    n_rows = 5000
    
    # Sample journals
    journals = ['Nature', 'Science', 'The Lancet', 'JAMA', 'BMJ', 'NEJM', 
                'PLOS One', 'Elsevier Journal', 'Springer Journal', 'IEEE Transactions']
    
    # Generate sample data
    data = {
        'title': [f'COVID-19 Research Paper {i}' for i in range(n_rows)],
        'abstract': [f'This is an abstract for paper {i} about COVID-19 and related research.' for i in range(n_rows)],
        'authors': [f'Author {i}, Coauthor {i}' for i in range(n_rows)],
        'journal': np.random.choice(journals, n_rows),
        'publish_time': np.random.choice(pd.date_range('2019-12-01', '2022-12-31'), n_rows),
        'url': [f'https://example.com/paper/{i}' for i in range(n_rows)]
    }
    
    df = pd.DataFrame(data)
    
    # Add some null values to simulate real data
    df.loc[df.sample(frac=0.1).index, 'abstract'] = np.nan
    df.loc[df.sample(frac=0.05).index, 'journal'] = np.nan
    
    return df

df = load_data()

# Add year column for analysis
df['year'] = pd.DatetimeIndex(df['publish_time']).year
df['month'] = pd.DatetimeIndex(df['publish_time']).month

# Sidebar filters
st.sidebar.subheader("Filter Data")
year_range = st.sidebar.slider(
    "Select Year Range",
    min_value=int(df['year'].min()),
    max_value=int(df['year'].max()),
    value=(2020, 2022)
)

selected_journals = st.sidebar.multiselect(
    "Select Journals",
    options=df['journal'].dropna().unique(),
    default=df['journal'].dropna().unique()[:3]
)

# Filter data based on selections
filtered_df = df[
    (df['year'] >= year_range[0]) & 
    (df['year'] <= year_range[1]) &
    (df['journal'].isin(selected_journals) if selected_journals else True)
]

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<h2 class="section-header">Data Overview</h2>', unsafe_allow_html=True)
    st.write(f"Displaying {len(filtered_df)} of {len(df)} papers")
    
    # Show basic statistics
    st.subheader("Basic Statistics")
    stats_col1, stats_col2, stats_col3 = st.columns(3)
    
    with stats_col1:
        st.metric("Total Papers", len(filtered_df))
    
    with stats_col2:
        st.metric("Journals Represented", filtered_df['journal'].nunique())
    
    with stats_col3:
        st.metric("Time Span", f"{filtered_df['year'].min()} - {filtered_df['year'].max()}")

with col2:
    st.markdown('<h2 class="section-header">Data Sample</h2>', unsafe_allow_html=True)
    st.dataframe(filtered_df[['title', 'journal', 'publish_time']].head(5))

# Tabs for different visualizations
tab1, tab2, tab3, tab4 = st.tabs([
    "Publications Over Time", 
    "Journal Analysis", 
    "Word Analysis", 
    "Data Details"
])

with tab1:
    st.markdown('<h2 class="section-header">Publications Over Time</h2>', unsafe_allow_html=True)
    
    # Time series analysis
    time_agg = st.radio("Aggregation", ["Yearly", "Monthly"], horizontal=True)
    
    if time_agg == "Yearly":
        time_data = filtered_df['year'].value_counts().sort_index()
        x_label = "Year"
    else:
        time_data = filtered_df.groupby(['year', 'month']).size()
        time_data.index = [f"{y}-{m:02d}" for y, m in time_data.index]
        x_label = "Month"
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time_data.index, time_data.values, marker='o', linewidth=2, markersize=8)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Number of Publications")
    ax.set_title("COVID-19 Publications Over Time")
    plt.xticks(rotation=45)
    st.pyplot(fig)

with tab2:
    st.markdown('<h2 class="section-header">Journal Analysis</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Top journals
        top_journals = filtered_df['journal'].value_counts().head(10)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(top_journals.index, top_journals.values, color='skyblue')
        ax.set_xlabel("Number of Publications")
        ax.set_title("Top Journals by Publication Count")
        st.pyplot(fig)
    
    with col2:
        # Journal by year
        if not filtered_df.empty:
            journal_year = pd.crosstab(filtered_df['journal'], filtered_df['year']).head(10)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            journal_year.plot(kind='bar', ax=ax)
            ax.set_title("Publications by Journal and Year")
            ax.set_ylabel("Number of Publications")
            plt.xticks(rotation=45)
            plt.legend(title='Year')
            st.pyplot(fig)

with tab3:
    st.markdown('<h2 class="section-header">Word Frequency Analysis</h2>', unsafe_allow_html=True)
    
    # Generate word cloud from titles
    text = " ".join(title for title in filtered_df['title'].dropna())
    
    if text:
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title('Word Cloud of Paper Titles')
        st.pyplot(fig)
        
        # Most common words
        words = re.findall(r'\w+', text.lower())
        stop_words = {'the', 'and', 'of', 'in', 'to', 'a', 'for', 'on', 'with', 'by', 'an', 'from'}
        filtered_words = [word for word in words if word not in stop_words and len(word) > 3]
        
        word_freq = Counter(filtered_words).most_common(15)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh([word[0] for word in word_freq], [word[1] for word in word_freq], color='lightcoral')
        ax.set_xlabel("Frequency")
        ax.set_title("Most Common Words in Titles")
        st.pyplot(fig)

with tab4:
    st.markdown('<h2 class="section-header">Data Details</h2>', unsafe_allow_html=True)
    
    st.subheader("Dataset Information")
    st.write(f"Total rows: {len(df)}")
    st.write(f"Total columns: {len(df.columns)}")
    
    st.subheader("Column Information")
    col_info = pd.DataFrame({
        'Column': df.columns,
        'Non-Null Count': df.notnull().sum(),
        'Data Type': df.dtypes
    })
    st.dataframe(col_info)
    
    st.subheader("Missing Values")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    
    if len(missing) > 0:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(missing.index, missing.values, color='orange')
        ax.set_ylabel("Number of Missing Values")
        ax.set_title("Missing Values by Column")
        plt.xticks(rotation=45)
        st.pyplot(fig)
    else:
        st.write("No missing values in the dataset.")

# Footer
st.markdown("---")
st.markdown("### About This App")
st.markdown("""
This application demonstrates the analysis of the CORD-19 dataset, which contains metadata 
about COVID-19 research papers. In a real implementation, you would replace the sample data 
with the actual metadata.csv file from the CORD-19 dataset.
""")

st.markdown("**Note:** This app uses simulated data for demonstration purposes.")