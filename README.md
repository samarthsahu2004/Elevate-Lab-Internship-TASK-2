# Elevate-Lab-Internship-TASK-2
Titanic Survival Exploratory Data Analysis (EDA)

📖 Overview
This project performs a detailed Exploratory Data Analysis (EDA) on the classic Titanic dataset. The goal is to uncover patterns, identify key factors influencing passenger survival, and derive insights through statistical summaries and data visualizations. This analysis serves as a foundational step before any machine learning modeling.

📊 Dataset
The project uses the Titanic Dataset available on Kaggle. It was downloaded programmatically using the kagglehub library.

The dataset contains passenger information like age, gender, passenger class, fare, and whether they survived the disaster.

Source: Titanic - Machine Learning from Disaster

Provider: Yasser H. on Kaggle Datasets

🛠️ Technologies & Libraries Used
This analysis is conducted using Python 3 and the following libraries:

kagglehub: For downloading the dataset directly from Kaggle.

pandas: For data manipulation and analysis.

numpy: For numerical operations.

matplotlib: For creating static plots.

seaborn: For enhanced statistical data visualization.

plotly: For creating interactive visualizations.

🚀 Setup & Usage
To run this analysis on your local machine, follow these steps:

1. Clone the repository (or download the files):

Bash

git clone https://github.com/your-username/titanic-eda.git
cd titanic-eda
2. Install the required libraries:
Make sure you have Python installed. Then, install the dependencies using pip.

Bash

pip install pandas numpy matplotlib seaborn plotly kagglehub
(Note: You may need to configure your Kaggle API credentials for kagglehub to work. Follow the instructions on the Kaggle website).

3. Run the analysis script:
Execute the main Python file from your terminal. The script will automatically download the data and generate the analysis plots.

Bash

python titanic_eda.py
The script will print summaries to the console and display a series of plots. Interactive plots from Plotly may open in your web browser.

📈 Key Findings from the EDA
The analysis revealed several key factors that strongly correlated with a passenger's chance of survival:

Gender was a critical factor: 🚺 Females had a significantly higher survival rate (over 74%) compared to males (under 19%), reflecting the "women and children first" protocol.

Socio-economic status mattered: 🎟️ 1st Class passengers had a much higher survival rate (~63%) than 3rd Class passengers (~24%).

Age played a role: 👶 Children and infants had a higher probability of survival than other age groups. A large number of passengers between 20-40 years old did not survive.

Family size influenced survival: Passengers traveling with 1-3 family members had a better survival rate than those traveling alone or with larger families.

Port of Embarkation: Passengers who embarked from Cherbourg ('C') had a higher survival rate compared to those from Southampton ('S') and Queenstown ('Q'), likely due to a higher proportion of 1st class passengers from that port.

This EDA provides a solid foundation for feature engineering and building a predictive model to determine passenger survival.
