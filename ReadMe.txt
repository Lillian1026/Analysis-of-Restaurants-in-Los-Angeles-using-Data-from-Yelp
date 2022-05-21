The main_program.py is the main program which collects, cleans, and analyzes all the relevant data in this project.

Requirements:
To successfully run this program, please make sure you have installed the following packages:
-pandas
-beautifulsoup4
-requests
-sys
-seaborn
-matplotlib
-scipy
-statsmodels
-langdetect
-nltk
-sklearn
-wordcloud
You can use pip install package_name to install them.
Also, please unzip the Restaurant_health_score.csv.zip and make sure the document Restaurant_health_score.csv is on the same path as the main_program.py.

Running:
You have three ways to run the main_program.py file in the command line:
- Input ./main_program.py: The program will print the complete datasets as rows of data.
- Input ./main_program.py --scrape N: The program will print the first N entries of the datasets.
- Input ./main_program.py --static: The program will save two datasets (Yelp_restaurant_USC.csv and Yelp_review.csv) onto the current path. 

Expected Output Files:
If you use ./main_program.py --static to run this program, you will get overall 4 datasets (except the Restaurant_health_score.csv dataset) in the current path: Yelp_restaurant_USC.csv, Yelp_review.csv, HealthScore_Clean_Data.csv, and Score_add.csv.
If you use two other ways to run this program, you will get overall 2 datasets (except the Restaurant_health_score.csv dataset) in the current path: HealthScore_Clean_Data.csv and Score_add.csv.
All the statistical results and graphs shown in the project description will be displayed.

Potential Problems and Solutions:
-Since Yelp has a restriction on the entry number, if the program gets stuck or does not work, please reduce the number of restaurants collected every time or change to another IP to try again. For example, you can change the res_df['URL'] to res_df['URL'][200:501] to control the number of collections every time.
-The code for scraping data on Yelp needs to be changed if the web structure of Yelp is changed.
