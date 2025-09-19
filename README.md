CORD-19 Data Explorer
A Streamlit application for exploring and visualizing the CORD-19 dataset, which contains metadata about COVID-19 research papers. This application provides interactive visualizations and analysis tools to understand trends in COVID-19 research publications.

Features
Interactive Data Exploration: Filter data by year range and journal selection

Time Series Analysis: Visualize publication trends over time (yearly or monthly)

Journal Analysis: Identify top journals publishing COVID-19 research

Word Frequency Analysis: Generate word clouds and frequency charts from paper titles

Data Quality Assessment: View missing values and dataset statistics

Responsive Design: Works on desktop and mobile devices

Installation
Clone or download this repository

Install the required dependencies:

bash
pip install streamlit pandas matplotlib seaborn wordcloud
Usage
Place your metadata.csv file from the CORD-19 dataset in the project directory

Run the application:

bash
streamlit run cord19_explorer.py
Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

Data Source
This application is designed to work with the CORD-19 dataset metadata file (metadata.csv). You can download the dataset from:

Allen Institute for AI - CORD-19

Application Structure
The application is organized into several sections:

Data Overview: Shows basic statistics and a sample of the data

Publications Over Time: Visualizes publication trends with interactive time aggregation

Journal Analysis: Displays top journals and their publication patterns

Word Frequency Analysis: Generates word clouds and common word frequency charts

Data Details: Provides information about the dataset structure and missing values

Customization
To adapt this application for your specific needs:

Modify the load_data() function to handle your specific data format

Adjust the visualizations in each tab section

Add new analysis features by creating additional tabs or sections

Customize the color scheme and styling in the CSS section

Notes
The current implementation uses simulated data for demonstration purposes

To use with real data, replace the load_data() function with code that loads your actual metadata.csv file

The application handles missing values and provides visualizations of data quality issues

Dependencies
streamlit

pandas

matplotlib

seaborn

wordcloud

numpy

License
This project is open source and available under the MIT License.

