# British Airways Virtual Job Simulation

This project focused on exploring the critical role of data science in British Airways' success. The goal was to collect, analyze, and model customer review data to uncover insights and build a predictive model for understanding factors influencing customer buying behavior.

## Objectives

- **Data Collection**: Scrape customer reviews from Skytrax to gather as much data as possible for the analysis.
- **Data Cleaning**: Clean the raw, messy data to prepare it for further analysis.
- **Data Analysis & Insights**: Use techniques like topic modeling, sentiment analysis, and word clouds to explore and understand the data.
- **Predictive Modeling**: Build a machine learning model to predict customer booking behavior based on the data.

## Approach

### 1. **Scrape Data from the Web**
   - The first step in this simulation was to scrape customer reviews about British Airways from Skytrax.
   - The focus was on gathering as much data as possible to improve the accuracy of the analysis.

### 2. **Analyze Data & Present Insights**
   - Once the data was collected, it was in raw text form and needed to be cleaned.
   - The cleaning process included removing unnecessary words, symbols, and stop words to make the data more suitable for analysis.
   - After cleaning, I used techniques such as:
     - **Topic Modeling** to identify key themes within the reviews.
     - **Sentiment Analysis** to gauge customer sentiment (positive, neutral, negative).
     - **Word Clouds** to visualize the most frequent terms used in the reviews.

### 3. **Explore and Prepare the Dataset**
   - To better understand the dataset, I explored it using a "Getting Started" Jupyter Notebook.
   - This included reviewing different columns, basic statistics, and visualizing trends.
   - I also worked on creating new features to improve the predictive power of the machine learning model.

### 4. **Train a Machine Learning Model**
   - The final step involved training a machine learning model to predict whether a customer would make a booking based on the available data.
   - I used a **RandomForest algorithm** for this task, as it is useful for understanding how each variable contributes to the model’s predictive power.

## Tools and Technologies Used

- **Python**: For data processing and modeling.
- **Jupyter Notebook**: For data exploration and analysis.
- **Pandas & NumPy**: For data manipulation and preparation.
- **Scikit-learn**: For building the machine learning model.
- **Matplotlib & Seaborn**: For data visualization.
- **NLTK & SpaCy**: For natural language processing and text analysis.

## Conclusion

This simulation allowed me to explore how data science can help British Airways improve its understanding of customer behavior and ultimately make better business decisions. By analyzing customer reviews and building a predictive model, the insights gained can help the marketing team develop more targeted strategies for customer engagement and improve overall business performance.
