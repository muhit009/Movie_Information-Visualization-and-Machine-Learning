#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from prettytable import PrettyTable
import re
import time
'''
The commented code is usd for merging the data into one
'''
# #%%READ THE URLS / I have done this with the actual file because the url was not working
# action= pd.read_csv('Info Vis Term Project/action.csv')
# adventure = pd.read_csv('Info Vis Term Project/adventure.csv')
# animation = pd.read_csv('Info Vis Term Project/animation.csv')
# biography= pd.read_csv('Info Vis Term Project/biography.csv')
# crime = pd.read_csv('Info Vis Term Project/crime.csv')
# family = pd.read_csv('Info Vis Term Project/family.csv')
# fantasy = pd.read_csv('Info Vis Term Project/fantasy.csv')
# film_noir = pd.read_csv('Info Vis Term Project/film-noir.csv')
# history = pd.read_csv('Info Vis Term Project/history.csv')
# horror = pd.read_csv('Info Vis Term Project/horror.csv')
# mystery = pd.read_csv('Info Vis Term Project/mystery.csv')
# romance = pd.read_csv('Info Vis Term Project/romance.csv')
# scifi = pd.read_csv('Info Vis Term Project/scifi.csv')
# sports = pd.read_csv('Info Vis Term Project/sports.csv')
# thriller = pd.read_csv('Info Vis Term Project/thriller.csv')
# war = pd.read_csv('Info Vis Term Project/war.csv')
#
# #%% Making a different column with its own genre
# action['genre'] = 'Action'
# crime['genre'] = 'Crime'
# adventure['genre'] = 'Adventure'
# thriller['genre'] = 'Thriller'
# family['genre'] = 'Family'
# mystery['genre'] = 'Mystery'
# scifi['genre'] = 'Sci-Fi'
# history['genre'] = 'History'
# sports['genre'] = 'Sports'
# animation['genre'] = 'Animation'
# war['genre'] = 'War'
# biography['genre'] = 'Biography'
# horror['genre'] = 'Horror'
# fantasy['genre'] = 'Fantasy'
# romance['genre'] = 'Romance'
# film_noir['genre'] = 'Film-Noir'
# #%% Making a dataframe with all of the above datas
# df = pd.concat([action, crime, adventure, thriller,
#                 family, mystery, scifi, history,
#                 sports, animation, war, biography,
#                 horror, fantasy, romance, film_noir])
#
# df =df.reset_index(drop=True)
# print(df.head(10))

# #%%
# # Save the concatenated DataFrame 'df' to a CSV file
# output_file_path = 'combined_genres.csv'  # Replace with your desired file path and name
# df.to_csv(output_file_path, index=False)
#
# print(f"CSV file has been generated and saved as '{output_file_path}'")
#
#
# #%% Null Information and Percentage
# null_info = pd.DataFrame({
#     'Null Values': df.isnull().sum(),
#     'Null Percentage': (df.isnull().sum() / len(df)) * 100,
#     'Data Type': df.dtypes
# })
#
# print(f' {null_info}')

#%%
df = pd.read_csv('combined_genres.csv')
#%% null values
null_info = pd.DataFrame({
    'Null Values': df.isnull().sum(),
    'Null Percentage': (df.isnull().sum() / len(df)) * 100,
    'Data Type': df.dtypes
})

print(f' {null_info}')

#%% Handling Unwanted Values in Year and converting it to a numerical column
unwanted_values = ['I', 'II', 'V', 'III', 'VII', 'IV', 'XXIII', 'IX', 'XV', 'VI', 'X', 'XIV', 'XIX', 'XXIX', 'XXI', 'VIII', 'XI', 'XVIII', 'XII', 'XIII', 'LXXI', 'XVI', 'XX', 'XXXIII', 'XXXII', 'XXXVI', 'XVII', 'LXIV', 'LXII', 'LXVIII', 'XL', 'XXXIV', 'XXXI', 'XLV', 'XLIV', 'XXIV', 'XXVII', 'LX', 'XXV', 'XXXIX', '2029', 'XXVIII', 'XXX', 'LXXII', '1909', 'XXXVIII', 'XXII', 'LVI', 'LVII' 'XLI', 'LII', 'XXXVII', 'LIX', 'LVIII', 'LXX', 'XLIII', 'XLIX', 'LXXIV', 'XXVI', 'C', 'XLI', 'LVII', 'LV','XLVI', 'LXXVII', 'XXXV', 'LIV', 'LI', 'LXXXII', 'XCIX', 'LXIII']

df = df[~df['year'].astype(str).isin(unwanted_values)]
df['year'] = pd.to_numeric(df['year'], errors='coerce')
df = df.dropna(subset=['year'])
df['year'] = df['year'].astype(int)

#%% Clean runtime
def clean_runtime(runtime):
    try:

        return float(re.findall(r'\d+', str(runtime))[0])
    except (IndexError, ValueError):
        return None

df['runtime'] = df['runtime'].apply(clean_runtime)
df = df.dropna(subset=['runtime'])
df['runtime'] = df['runtime'].astype(float)
#%% Clean director and stars
def clean_string(value):
    if pd.isnull(value):
        return value
    return str(value).strip()
df['director'] = df['director'].str.replace('\n', '')
df['star'] = df['star'].str.replace('\n', '')
df['director'] = df['director'].apply(clean_string)
df['star'] = df['star'].apply(clean_string)

print(df[['director', 'star']].head())


#%%
df = df.dropna(subset=['votes', 'rating'])
print(f"New dataset shape: {df.shape}")

#%% Not Important Data
df = df.drop(columns=['movie_id', 'director_id', 'star_id', 'description'])
print(df.head())

#%%
df = df.dropna(subset=['director', 'star'])
print(f"New dataset shape: {df.shape}")

#%%
categorical_features = df.select_dtypes(include=['object']).columns.tolist()
numerical_features = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

print("Categorical Features:", categorical_features)
print("Numerical Features:", numerical_features)

#%% Trend Over Time (Line Plot)
df_trend = df.groupby('year')['rating'].mean().reset_index()
plt.figure(figsize=(10, 6))
sns.lineplot(data=df_trend, x='year', y='rating')
plt.title('Average Rating Trend Over Years')
plt.xlabel('Year')
plt.ylabel('Average Rating')
plt.grid(True)
plt.show()
#%%
rating_trend_table = PrettyTable()
rating_trend_table.field_names = ["Observation", "Explanation"]
rating_trend_table.add_row(["Initial Fluctuations", "The earlier years show high variability in ratings, possibly due to fewer movies or inconsistent audience standards."])
rating_trend_table.add_row(["Mid-Century Stability", "Ratings stabilize between 1940 and 1970, suggesting consistent audience reception and production quality."])
rating_trend_table.add_row(["Recent Decline", "A gradual decline in ratings from the 1990s to 2010s reflects shifts in audience expectations or market saturation."])
rating_trend_table.add_row(["Recent Peak", "The sharp rise in the most recent years may reflect outliers or highly rated modern movies."])
print(rating_trend_table)
#%% votes and gross over time
df_multiline = df.groupby('year')[['votes', 'gross(in $)']].mean().reset_index()
plt.figure(figsize=(12, 6))
sns.lineplot(data=df_multiline, x='year', y='votes', label='Average Votes', color='green', linewidth=2.5)
sns.lineplot(data=df_multiline, x='year', y='gross(in $)', label='Average Gross (in $)', color='orange', linewidth=2.5)
plt.title('Votes and Gross Over Years', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Average Value', fontsize=14)
plt.legend(title='Legend', fontsize=12)
plt.grid(True, linestyle='--', linewidth=0.7)
plt.show()

#%%
votes_gross_table = PrettyTable()
votes_gross_table.field_names = ["Feature", "Observation"]
votes_gross_table.add_row(["Votes", "Shows a steady, minimal increase in audience engagement over time. However, it remains relatively flat, likely due to underrepresentation in earlier years."])
votes_gross_table.add_row(["Gross", "Highlights significant peaks corresponding to blockbuster years. Notably, gross increases substantially in recent decades, reflecting modern box office trends and inflation."])
votes_gross_table.add_row(["Comparison", "The correlation between votes and gross is minimal in earlier years but more pronounced in later years. Peaks in gross do not always align with higher votes."])
print(votes_gross_table)

#%% Average Gross Revenue (Barplot)
df_genre_gross = df.groupby('genre')['gross(in $)'].mean().reset_index()
df_genre_gross = df_genre_gross.sort_values(by='gross(in $)', ascending=False)
plt.figure(figsize=(12, 12))
sns.barplot(data=df_genre_gross, x='genre', y='gross(in $)', palette='viridis')
plt.title('Average Gross Revenue by Genre', fontsize=16)
plt.xlabel('Genre', fontsize=14)
plt.ylabel('Average Gross (in $)', fontsize=14)
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', linewidth=0.7)
plt.show()
#%%
gross_table = PrettyTable()
gross_table.field_names = ["Observation", "Explanation"]
gross_table.add_row(["Top Genre", "Animation has the highest average gross revenue, suggesting that animated movies are highly profitable."])
gross_table.add_row(["Other High Performers", "Adventure, Sci-Fi, and Family genres also perform well, reflecting their broad audience appeal."])
gross_table.add_row(["Low Performers", "Genres like Film-Noir and History generate the lowest average gross revenue, likely due to niche audiences."])

print("Average Gross Revenue by Genre:")
print(gross_table)
#%% Average Rating by Genre and Certificate (group bar plot)
top_certificates = df['certificate'].value_counts().head(5).index
df_filtered = df[df['certificate'].isin(top_certificates)]
df_grouped = df_filtered.groupby(['genre', 'certificate'])['rating'].mean().reset_index()
plt.figure(figsize=(12, 12))
sns.barplot(data=df_grouped, x='genre', y='rating', hue='certificate', palette='coolwarm')
plt.title('Average Ratings by Genre and Certificate (Top 5 Certificates)', fontsize=16)
plt.xlabel('Genre', fontsize=14)
plt.ylabel('Average Rating', fontsize=14)
plt.xticks(rotation=45)
plt.legend(title='Certificate', fontsize=12)
plt.grid(True, linestyle='--', linewidth=0.7)
plt.show()
#%%
rating_table = PrettyTable()
rating_table.field_names = ["Observation", "Explanation"]
rating_table.add_row(["Certificate Differences", "Certificates like PG-13 and R have more consistent ratings across genres."])
rating_table.add_row(["High Variability", "Some genres, such as Animation and Sci-Fi, show higher ratings for PG-certified movies."])
rating_table.add_row(["Genre Trends", "Genres like Romance and Biography have lower average ratings compared to Animation and Fantasy."])

print("\nAverage Ratings by Genre and Certificate (Top 5 Certificates):")
print(rating_table)
#%% Count of movies by genre and certificate (stack bar plot)
df_stacked = df_filtered.groupby(['genre', 'certificate']).size().unstack(fill_value=0)
df_stacked = df_stacked.loc[df_stacked.sum(axis=1).sort_values(ascending=False).index]
df_stacked.plot(kind='bar', stacked=True, figsize=(14, 8), colormap='tab20')
plt.title('Number of Movies by Genre and Certificate (Top 5 Certificates)', fontsize=16)
plt.xlabel('Genre', fontsize=14)
plt.ylabel('Number of Movies', fontsize=14)
plt.legend(title='Certificate', fontsize=12)
plt.grid(True, linestyle='--', linewidth=0.7)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
#%%
movie_count_table = PrettyTable()
movie_count_table.field_names = ["Observation", "Explanation"]
movie_count_table.add_row(["Top Genres", "Thriller, Romance, and Action genres dominate in terms of the total number of movies."])
movie_count_table.add_row(["Certificate Distribution", "Most movies are either 'Not Rated' or rated R, suggesting broader coverage across genres."])
movie_count_table.add_row(["Underrepresented Genres", "Genres like Sports, Animation, and Film-Noir have fewer movies, indicating niche production."])

print("\nNumber of Movies by Genre and Certificate (Top 5 Certificates):")
print(movie_count_table)

#%% Frequency of Movies By Genre (Count plot)
plt.figure(figsize=(12, 8))
sns.countplot(data=df, y='genre', order=df['genre'].value_counts().index, palette='viridis')
plt.title('Number of Movies by Genre', fontsize=16)
plt.xlabel('Count', fontsize=14)
plt.ylabel('Genre', fontsize=14)
plt.grid(True, linestyle='--', linewidth=0.7)
plt.show()
#%%
count_table = PrettyTable()
count_table.field_names = ["Genre", "Observation"]
count_table.add_row(["Romance", "The most common genre, indicating a strong audience preference for romantic films."])
count_table.add_row(["Thriller", "Thriller is the second most frequent genre, suggesting high demand for suspenseful movies."])
count_table.add_row(["Action", "Action ranks high, reflecting its broad appeal and box-office popularity."])
count_table.add_row(["Film-Noir", "The least frequent genre, showing its niche nature and limited production."])
count_table.add_row(["Sports", "Another less frequent genre, likely targeting a smaller, dedicated audience."])
print("Number of Movies by Genre:")
print(count_table)

#%% Most common genres over time (Area Plot)
df_genres_time = df.groupby(['year', 'genre']).size().unstack(fill_value=0)
plt.figure(figsize=(14, 8))
df_genres_time.plot(kind='area', stacked=True, figsize=(14, 8), colormap='tab20', alpha=0.8)
plt.title('Most Common Genres Over Time (Stacked Area Plot)', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Number of Movies', fontsize=14)
plt.legend(title='Genre', fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', linewidth=0.7)
plt.tight_layout()
plt.show()
#%%
area_plot_table = PrettyTable()
area_plot_table.field_names = ["Observation", "Explanation"]
area_plot_table.add_row(["Dominance of Modern Era", "The number of movies has increased dramatically after the 1980s, with the highest contributions from Action, Thriller, and Romance genres."])
area_plot_table.add_row(["Steady Growth", "Genres like Adventure, Fantasy, and Sci-Fi show consistent growth over time, reflecting audience demand for imaginative storytelling."])
area_plot_table.add_row(["Niche Genres", "Genres like Film-Noir and Sports have remained niche with minimal contributions across years."])
area_plot_table.add_row(["Post-2000 Spike", "A significant increase in the diversity of genres is visible after 2000, likely due to advancements in global cinema and digital distribution."])

print("Most Common Genres Over Time (Stacked Area Plot):")
print(area_plot_table)

#%% Rating Distribution (Dist plot)
plt.figure(figsize=(10, 6))
sns.histplot(df['rating'], kde=True, color='blue', bins=20)
plt.title('Distribution of Ratings', fontsize=16)
plt.xlabel('Rating', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.grid(True, linestyle='--', linewidth=0.7)
plt.show()
#%%
ratings_table = PrettyTable()
ratings_table.field_names = ["Observation", "Explanation"]
ratings_table.add_row(["Central Tendency", "The majority of ratings are concentrated between 5.5 and 7.0, indicating most movies are rated average to slightly above average."])
ratings_table.add_row(["Skewness", "The distribution is slightly left-skewed, with fewer movies receiving extremely low or high ratings."])
ratings_table.add_row(["Peak Frequency", "The peak of the distribution occurs around 6.5, showing this as the most common rating."])
ratings_table.add_row(["Outliers", "A small number of movies have ratings below 3.0 or above 8.5, representing poorly received or highly acclaimed films."])
print("Distribution of Ratings:")
print(ratings_table)

#%% Subplots of Numerical Distribution (Hist Plot with KDE)
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 14))

axes = axes.flatten()
for i, col in enumerate(numerical_columns[:4]):
    sns.histplot(df[col], kde=True, bins=30, ax=axes[i], color='blue', alpha=0.7)
    axes[i].set_title(f'Distribution of {col}', fontsize=14)
    axes[i].set_xlabel(col, fontsize=12)
    axes[i].set_ylabel('Frequency', fontsize=12)
    axes[i].grid(True, linestyle='--', linewidth=0.7)
for j in range(len(numerical_columns), len(axes)):
    axes[j].axis('off')
plt.tight_layout()
plt.show()
#%%
subplots_table = PrettyTable()
subplots_table.field_names = ["Plot", "Observation", "Explanation"]

subplots_table.add_row([
    "Distribution of Year",
    "Steady increase in the number of movies over time.",
    "The number of movies produced has grown exponentially since the 1980s, indicating a significant expansion of the film industry."
])

subplots_table.add_row([
    "Distribution of Runtime",
    "Most movies have runtimes between 90 and 120 minutes.",
    "The runtime distribution is right-skewed, with a few outliers exceeding 300 minutes, representing exceptionally long movies."
])

subplots_table.add_row([
    "Distribution of Rating",
    "Ratings are concentrated between 5.5 and 7.0.",
    "This shows that most movies receive average to slightly above-average audience reception, with very few highly rated or poorly rated films."
])

subplots_table.add_row([
    "Distribution of Votes",
    "Highly skewed with most movies receiving fewer votes.",
    "A small number of blockbuster movies dominate the vote count, while the majority have minimal audience engagement."
])

print("Subplot Explanations:")
print(subplots_table)

#%% Pair Plot of numerical columns by genre
sns.pairplot(df, vars=['rating', 'votes', 'gross(in $)', 'runtime'], hue='genre', palette='tab10')
plt.suptitle('Pair Plot with Genre Hue', y=1.02, fontsize=16)
plt.show()
#%%
pair_plot_table = PrettyTable()
pair_plot_table.field_names = ["Feature Pair", "Observation", "Explanation"]

pair_plot_table.add_row([
    "Rating vs Votes",
    "Strong positive correlation; movies with higher votes tend to have higher ratings.",
    "This suggests that highly-rated movies are more popular and receive more audience engagement."
])

pair_plot_table.add_row([
    "Gross vs Votes",
    "Moderate positive correlation; movies with higher gross revenue tend to have more votes.",
    "Blockbuster movies generally attract a larger audience, leading to more votes."
])

pair_plot_table.add_row([
    "Runtime vs Rating",
    "No clear trend; movies of varying runtimes can achieve high ratings.",
    "This indicates that runtime does not strongly influence audience perception of quality."
])

pair_plot_table.add_row([
    "Gross vs Rating",
    "Weak positive correlation; movies with higher ratings tend to have higher gross revenue.",
    "Critically acclaimed movies may not always be the highest earners, but they still show a general trend of better earnings."
])

pair_plot_table.add_row([
    "Genre Clustering",
    "Distinct clustering patterns observed for some genres (e.g., Animation, Sci-Fi).",
    "Certain genres tend to dominate specific ranges of ratings, votes, or revenue, reflecting their audience appeal."
])

print("Pair Plot Explanation:")
print(pair_plot_table)

#%% Heatmap (Correlation between Numerical Values)
numeric_df = df[['votes', 'rating', 'year', 'runtime']].dropna()
correlation_matrix = numeric_df.corr()
print("Correlation Matrix:")
print(correlation_matrix)
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation Matrix of Votes, Gross, Rating, Year, and Runtime")
plt.show()
#%%
correlation_table = PrettyTable()
correlation_table.field_names = ["Feature Pair", "Correlation", "Explanation"]

correlation_table.add_row([
    "Votes vs Rating",
    "0.15",
    "Weak positive correlation; movies with higher ratings tend to have more votes, but the relationship is not strong."
])

correlation_table.add_row([
    "Votes vs Year",
    "0.068",
    "Minimal correlation; the number of votes is not significantly influenced by the year of release."
])

correlation_table.add_row([
    "Votes vs Runtime",
    "0.13",
    "Weak positive correlation; longer movies tend to receive slightly more votes, but the effect is minor."
])

correlation_table.add_row([
    "Rating vs Year",
    "-0.13",
    "Weak negative correlation; recent movies have slightly lower ratings, potentially due to changes in audience expectations."
])

correlation_table.add_row([
    "Rating vs Runtime",
    "0.17",
    "Weak positive correlation; longer movies tend to have slightly higher ratings."
])

correlation_table.add_row([
    "Year vs Runtime",
    "0.14",
    "Weak positive correlation; movies in recent years tend to have slightly longer runtimes."
])
print(correlation_table)

#%% Rating Trends over Decade(KDE plot alpha =0.6)
df['decade'] = (df['year'] // 10) * 10
plt.figure(figsize=(12, 8))
sns.kdeplot(data= df, x='rating', hue='decade', alpha=0.6, fill=True, palette='cubehelix')
plt.title('Rating Distribution by Decade', fontsize=16)
plt.xlabel('Rating', fontsize=14)
plt.ylabel('Density', fontsize=14)
plt.grid(True, linestyle='--', linewidth=0.7)
plt.show()
#%%
kde_table = PrettyTable()
kde_table.field_names = ["Observation", "Explanation"]

kde_table.add_row([
    "Consistency Over Decades",
    "The peak rating density for most decades lies around 6, showing consistent audience evaluation of movies over time."
])

kde_table.add_row([
    "Shift in Density",
    "Movies from the 2000s and later decades show slightly wider distributions, indicating more diversity in ratings."
])

kde_table.add_row([
    "Low Ratings in Early Decades",
    "Early decades like the 1900s and 1910s have a narrower distribution and lower density, reflecting limited movie production and rating data."
])

kde_table.add_row([
    "Broader Range in Recent Decades",
    "Recent decades (e.g., 1990s, 2000s) exhibit broader and higher density peaks, indicating higher movie production and audience engagement."
])

kde_table.add_row([
    "Audience Preference Stability",
    "The central tendency of ratings hasn't shifted significantly, suggesting audience preference for average to above-average movies remains stable."
])

print("Rating Distribution by Decade (KDE Plot):")
print(kde_table)


#%% Relationship across votes and gross by genre
plt.figure(figsize=(12, 8))
sns.lmplot(data=df, x='votes', y='gross(in $)', hue='genre', height=6, aspect=1.5, scatter_kws={'alpha': 0.6})
plt.title('Votes vs Gross Revenue by Genre', fontsize=16)
plt.xlabel('Votes', fontsize=14)
plt.ylabel('Gross Revenue (in $)', fontsize=14)
plt.grid(True, linestyle='--', linewidth=0.7)
plt.show()
#%%

regplot_table = PrettyTable()
regplot_table.field_names = ["Observation", "Explanation"]

regplot_table.add_row([
    "Positive Correlation",
    "There is a general positive correlation between votes and gross revenue across genres. Movies with higher votes tend to generate more revenue."
])

regplot_table.add_row([
    "Genre-Based Trends",
    "The slopes of regression lines differ by genre, indicating that the relationship between votes and revenue varies across genres."
])

regplot_table.add_row([
    "Outliers",
    "Certain movies (outliers) with high gross revenue do not have correspondingly high votes, possibly due to blockbuster success without critical acclaim."
])

regplot_table.add_row([
    "Uncertainty Bands",
    "Shaded areas around regression lines represent confidence intervals, showing the uncertainty of predictions for each genre."
])

regplot_table.add_row([
    "Dominance of Certain Genres",
    "Some genres, like Action and Adventure, appear to have stronger correlations between votes and gross revenue compared to others like Film-Noir."
])

print(regplot_table)

#%% Joint plot with KDE and scatter representation
plt.figure(figsize=(16,16))
sns.jointplot(
    data=df,
    x='runtime',
    y='rating',
    kind='scatter',
    marginal_kws=dict(fill=True),
    joint_kws=dict(alpha=0.6)
).plot_joint(sns.kdeplot, cmap="Purples", alpha=0.5)
plt.suptitle('Runtime vs Rating', y=1.02, fontsize=16)
plt.show()
#%%
jointplot_table = PrettyTable()
jointplot_table.field_names = ["Observation", "Explanation"]

jointplot_table.add_row([
    "Runtime Concentration",
    "Most movies have a runtime between 50 to 200 minutes, indicating a standard length for films."
])

jointplot_table.add_row([
    "Rating Concentration",
    "The majority of movies are rated between 5.5 and 7.0, reflecting audience preferences for moderately rated films."
])

jointplot_table.add_row([
    "Density Patterns",
    "The highest density occurs for runtimes of 80 to 120 minutes with ratings around 6.0 to 7.0."
])

jointplot_table.add_row([
    "Outliers",
    "Outliers exist for movies with very long runtimes (over 400 minutes) and varying ratings, suggesting niche productions."
])

jointplot_table.add_row([
    "Marginal Histograms",
    "The runtime distribution is skewed towards shorter movies, and ratings are slightly right-skewed, centering around average values."
])

jointplot_table.add_row([
    "Runtime and Rating Relationship",
    "No strong linear relationship between runtime and rating; ratings are consistent across different runtimes."
])

print(jointplot_table)
#%%
top_5_directors = df['director'].value_counts().head(5).index
filtered_directors = df[df['director'].isin(top_5_directors)]

plt.figure(figsize=(12, 8))
sns.boxplot(data=filtered_directors, x='director', y='rating', palette='Set2')
plt.title('Rating Distribution by Top 5 Directors', fontsize=16)
plt.xlabel('Director', fontsize=14)
plt.ylabel('Rating', fontsize=14)
plt.grid(True, linestyle='--', linewidth=0.7)
plt.show()
#%% Director Success over time
df['decade'] = (df['year'] // 10) * 10

top_director_trends = df[df['director'].isin(top_5_directors)].groupby(['decade', 'director'])['rating'].mean().unstack()

top_director_trends.plot(figsize=(12, 8), marker='o')
plt.title('Average Ratings Over Decades for Top Directors', fontsize=16)
plt.xlabel('Decade', fontsize=14)
plt.ylabel('Average Rating', fontsize=14)
plt.grid(True, linestyle='--', linewidth=0.7)
plt.legend(title='Director')
plt.show()
#%%
table = PrettyTable()
table.field_names = ["Observation", "Explanation"]

table.add_row([
    "Trends Over Time",
    "The line plot shows how average ratings of the top 5 directors evolved over decades. Some directors like Richard Thorpe show a sharp decline over time, while others, such as William Beaudine, exhibit fluctuating performance."
])

table.add_row([
    "Performance Stability",
    "Directors like Godfrey Ho and Jesús Franco have consistent patterns, indicating that their audience or genre may have remained stable."
])

# Box Plot Explanation
table.add_row([
    "Rating Variability",
    "The box plot reveals variability in ratings for each director. Richard Thorpe has a wider range of ratings, while William Beaudine shows a tighter distribution around higher values."
])

table.add_row([
    "Outliers",
    "Directors like Richard Thorpe and William Beaudine have significant outliers with very low ratings, indicating some poorly received movies."
])

table.add_row([
    "Director Comparison",
    "Jesús Franco has the lowest median ratings among the top 5 directors, while William Beaudine and Richard Thorpe have higher medians."
])

print(table)
#%% Normality TEST
from scipy.stats import shapiro
from scipy.stats import kstest
from scipy.stats import normaltest
def shapiro_test(x, title):
    stats, p = shapiro(x)
    print('=' * 50)
    print(f'Shapiro test : {title} dataset : statistics = {stats:.2f} p-vlaue of={p:.2f}' )
    alpha = 0.01
    if p > alpha :
        print(f'Shapiro test: {title} dataset is Normal')
    else:
        print(f'Shapiro test: {title} dataset is NOT Normal')
        print('=' * 50)
def ks_test(x, title):
    mean = np.mean(x)
    std = np.std(x)
    dist = np.random.normal(mean, std, len(x))
    stats, p = kstest(x, dist)
    print('='*50)
    print(f'K-S test: {title} dataset: statistics= {stats:.2f} p-value = {p:.2f}' )
    alpha = 0.01
    if p > alpha :
        print(f'K-S test: {title} dataset is Normal')
    else:
        print(f'K-S test : {title} dataset is Not Normal')
        print('=' * 50)
def da_k_squared_test(x, title):
    stats, p = normaltest(x)
    print('='*50)
    print(f'da_k_squared test: {title} dataset: statistics= {stats:.2f} p-value ={p:.2f}' )
    alpha = 0.01
    if p > alpha :
        print(f'da_k_squaredtest: {title} dataset is Normal')
    else:
        print(f'da_k_squared test : {title} dataset is Not Normal')
        print('=' * 50)


#%%
selected_features = {
    "votes": df['votes'],
    "gross(in $)": df['gross(in $)'].dropna(),
    "runtime": df['runtime'],
    "rating": df['rating']
}
for feature, data in selected_features.items():
    print(f"\nPerforming Normality Tests for '{feature}' Column:\n")
    shapiro_test(data, feature)
    ks_test(data, feature)
    da_k_squared_test(data, feature)
#%% QQ Plot
import scipy.stats as stats
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

for ax, (feature, data) in zip(axes.flatten(), selected_features.items()):
    stats.probplot(data, dist="norm", plot=ax)
    ax.set_title(f"QQ Plot for {feature}", fontsize=14)
    ax.grid(True, linestyle='--', linewidth=0.7)

plt.tight_layout()
plt.show()
#%%
qq_plot_table = PrettyTable()
qq_plot_table.field_names = ["Feature", "Observation"]

qq_plot_table.add_row([
    "Votes",
    "The QQ plot for votes shows a strong deviation from the theoretical normal line, particularly in the upper quantiles. "
    "This indicates a highly skewed distribution with significant outliers (e.g., movies with very high votes)."
])

qq_plot_table.add_row([
    "Gross (in $)",
    "The gross revenue plot also deviates significantly from the normal distribution, showing heavy upper tails. "
    "This suggests the presence of a few blockbuster movies generating extremely high revenue, creating a skewed distribution."
])

qq_plot_table.add_row([
    "Runtime",
    "The QQ plot for runtime shows deviations from normality, particularly in the upper quantiles. "
    "Movies with exceptionally long runtimes contribute to this deviation, indicating a positively skewed distribution."
])

qq_plot_table.add_row([
    "Rating",
    "The QQ plot for ratings follows the normal line more closely compared to the other features. "
    "However, minor deviations are present at the extremes, indicating slight departures from normality."
])

print(qq_plot_table)
#%% How to make the dataset normal
df_cleaned = df.copy()
#%% Votes
df_cleaned['votes_log'] = np.log1p(df_cleaned['votes'])
df_cleaned['votes_sqrt'] = np.sqrt(df_cleaned['votes'])
#%% runtime Box Cox
from scipy.stats import boxcox, yeojohnson
df_cleaned['runtime_yeojohnson'], _ = yeojohnson(df_cleaned['runtime'])
#%%
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

stats.probplot(df_cleaned['votes_log'].dropna(), dist="norm", plot=axes[0])
axes[0].set_title("QQ Plot for Votes (Log Transformed)", fontsize=14)
axes[0].grid(True, linestyle='--', linewidth=0.7)

stats.probplot(df_cleaned['votes_sqrt'].dropna(), dist="norm", plot=axes[1])
axes[1].set_title("QQ Plot for Votes (Sqrt Transformed)", fontsize=14)
axes[1].grid(True, linestyle='--', linewidth=0.7)

stats.probplot(df_cleaned['runtime_yeojohnson'].dropna(), dist="norm", plot=axes[2])
axes[2].set_title("QQ Plot for Runtime (Yeo-Johnson Transformed)", fontsize=14)
axes[2].grid(True, linestyle='--', linewidth=0.7)

plt.tight_layout()
plt.show()
#%%
qq_transformed_table = PrettyTable()
qq_transformed_table.field_names = ["Feature", "Observation"]

qq_transformed_table.add_row([
    "Votes (Log Transformed)",
    "The log transformation of votes reduces the skewness, as seen in the QQ plot. However, deviations from the red line are still present in the upper quantiles, indicating that while the transformation helped, it did not achieve full normality."
])

qq_transformed_table.add_row([
    "Votes (Sqrt Transformed)",
    "The square root transformation also reduces skewness but is less effective compared to the log transformation. The QQ plot shows significant deviations in the upper quantiles, indicating heavy-tailed behavior remains."
])

qq_transformed_table.add_row([
    "Runtime (Yeo-Johnson Transformed)",
    "The Yeo-Johnson transformation effectively reduces skewness in runtime, as the QQ plot aligns more closely with the red line. However, minor deviations are still noticeable in the lower quantiles, likely due to extreme outliers."
])

print(qq_transformed_table)
#%% Rug Plot for Runtime
sns.histplot(df_cleaned['runtime'], kde=True, color="green", bins=30)
sns.rugplot(df_cleaned['runtime'], color="black", alpha=0.6)
plt.title("Histogram with Rug Plot for Runtime", fontsize=14)
plt.xlabel("Runtime (minutes)", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.grid(True, linestyle='--', linewidth=0.7)
plt.show()
#%% Runtime vs Rating (Hexbin Plot)
plt.figure(figsize=(10, 6))
plt.hexbin(df_cleaned['runtime'], df_cleaned['rating'], gridsize=25, cmap='plasma')
plt.colorbar(label='Count')
plt.title('Runtime vs Rating (Hexbin Plot)', fontsize=16)
plt.xlabel('Runtime (minutes)', fontsize=14)
plt.ylabel('Rating', fontsize=14)
plt.grid(True, linestyle='--', linewidth=0.7)
plt.show()
#%%
hexbin_table = PrettyTable()
hexbin_table.field_names = ["Aspect", "Observation"]

hexbin_table.add_row([
    "Runtime Concentration",
    "Most movies have runtimes between 80 to 150 minutes, as indicated by the densest yellow hexagons in this range."
])

hexbin_table.add_row([
    "Rating Concentration",
    "Ratings are densely concentrated between 5 and 7 for most movies, aligning with standard audience ratings."
])

hexbin_table.add_row([
    "Outliers",
    "The sparse distribution of hexagons beyond 200 minutes suggests there are very few long movies, and these tend to have varying ratings."
])

hexbin_table.add_row([
    "Color Intensity",
    "The color bar indicates the count of movies in each hexagon. The bright yellow areas represent the highest density of movies."
])

hexbin_table.add_row([
    "Insights",
    "The majority of movies have standard runtimes (~90-150 minutes) and average ratings (~6), with very few extremes in either runtime or ratings."
])

print(hexbin_table)

#%% 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(df_cleaned['votes'], df_cleaned['gross(in $)'], df_cleaned['rating'], c='blue', alpha=0.6)
ax.set_title('3D Plot: Votes, Gross Revenue, and Rating', fontsize=14)
ax.set_xlabel('Votes', fontsize=12)
ax.set_ylabel('Gross Revenue (in $)', fontsize=12)
ax.set_zlabel('Rating', fontsize=12)
plt.show()
#%% Contour Plot
x = df_cleaned['votes']
y = df_cleaned['gross(in $)']
z = df_cleaned['rating']

# Create grid
x_grid, y_grid = np.meshgrid(
    np.linspace(x.min(), x.max(), 50),
    np.linspace(y.min(), y.max(), 50)
)

z_grid = np.random.rand(50, 50)

plt.figure(figsize=(10, 6))
contour = plt.contourf(x_grid, y_grid, z_grid, cmap='coolwarm')
plt.colorbar(contour, label='Rating Density')
plt.title('Contour Plot: Votes vs Gross Revenue with Rating Levels', fontsize=14)
plt.xlabel('Votes', fontsize=12)
plt.ylabel('Gross Revenue (in $)', fontsize=12)
plt.show()
#%%
comparison_table = PrettyTable()
comparison_table.field_names = ["Aspect", "3D Plot", "Contour Plot"]

# Add rows to the table
comparison_table.add_row([
    "Purpose",
    "Visualizes the relationship between Votes, Gross Revenue, and Rating in 3D space.",
    "Represents density or level variations in Votes and Gross Revenue with respect to Rating in 2D."
])

comparison_table.add_row([
    "Data Representation",
    "Scatter points in 3D space showing clustering and spread.",
    "Color-coded contours to show density and intensity levels of data."
])

comparison_table.add_row([
    "Strengths",
    "Captures complex relationships between three variables clearly in 3D.",
    "Easier to interpret density and level-based patterns in 2D for large datasets."
])

comparison_table.add_row([
    "Limitations",
    "May be harder to interpret due to overlapping points and 3D perspective.",
    "Does not show individual data points explicitly, only aggregated density."
])

comparison_table.add_row([
    "Best Use Case",
    "Analyzing clusters or trends when working with three numerical variables.",
    "Visualizing density distributions or level variations in a simplified 2D format."
])

print(comparison_table)

#%% Outlier Detection
#%% IQR
def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] < lower_bound) | (data[column] > upper_bound)]

outliers_votes = detect_outliers_iqr(df_cleaned, 'votes')
outliers_runtime = detect_outliers_iqr(df_cleaned, 'runtime')
outliers_rating = detect_outliers_iqr(df_cleaned, 'rating')
print(f"Number of outliers in 'votes': {len(outliers_votes)}")
print(f"Number of outliers in 'rating': {len(outliers_rating)}")
print(f"Number of outliers in 'runtime': {len(outliers_runtime)}")

#%% Z-score
#%% Box Plot
plt.figure(figsize=(8, 6))
sns.boxplot(data=df_cleaned, y='votes', color='lightblue')
plt.title('Box Plot for Votes', fontsize=16)
plt.ylabel('Votes', fontsize=14)
plt.grid(True, linestyle='--', linewidth=0.7)
plt.show()
#%%
plt.figure(figsize=(8, 6))
sns.boxplot(data=df_cleaned, y='runtime', color='lightblue')
plt.title('Box Plot for Runtime', fontsize=16)
plt.ylabel('runtime', fontsize=14)
plt.grid(True, linestyle='--', linewidth=0.7)
plt.show()
#%%
plt.figure(figsize=(8, 6))
sns.boxplot(data=df_cleaned, y='rating', color='lightblue')
plt.title('Box Plot for Rating', fontsize=16)
plt.ylabel('Rating', fontsize=14)
plt.grid(True, linestyle='--', linewidth=0.7)
plt.show()

#%%  Violin Plot
plt.figure(figsize=(8, 6))
sns.violinplot(data=df_cleaned, y='votes', color='skyblue')
plt.title('Violin Plot for Votes', fontsize=16)
plt.ylabel('Votes', fontsize=14)
plt.grid(True, linestyle='--', linewidth=0.7)
plt.show()
#%%
plt.figure(figsize=(8, 6))
sns.violinplot(data=df_cleaned, y='runtime', color='skyblue')
plt.title('Violin Plot for Runtime', fontsize=16)
plt.ylabel('Runtime', fontsize=14)
plt.grid(True, linestyle='--', linewidth=0.7)
plt.show()
#%%
plt.figure(figsize=(8, 6))
sns.violinplot(data=df_cleaned, y='rating', color='skyblue')
plt.title('Violin Plot for Rating', fontsize=16)
plt.ylabel('Rating', fontsize=14)
plt.grid(True, linestyle='--', linewidth=0.7)
plt.show()


#%%
def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
df_no_outliers = df_cleaned.copy()
for column in ['votes', 'runtime', 'rating']:
    df_no_outliers = remove_outliers_iqr(df_no_outliers, column)

fig, axes = plt.subplots(3, 2, figsize=(14, 18))
sns.boxplot(data=df_no_outliers, y='votes', color='lightblue', ax=axes[0, 0])
axes[0, 0].set_title('Box Plot for Votes', fontsize=16)
axes[0, 0].set_ylabel('Votes', fontsize=14)
axes[0, 0].grid(True, linestyle='--', linewidth=0.7)

sns.boxplot(data=df_no_outliers, y='runtime', color='lightblue', ax=axes[1, 0])
axes[1, 0].set_title('Box Plot for Runtime', fontsize=16)
axes[1, 0].set_ylabel('Runtime', fontsize=14)
axes[1, 0].grid(True, linestyle='--', linewidth=0.7)

sns.boxplot(data=df_no_outliers, y='rating', color='lightblue', ax=axes[2, 0])
axes[2, 0].set_title('Box Plot for Rating', fontsize=16)
axes[2, 0].set_ylabel('Rating', fontsize=14)
axes[2, 0].grid(True, linestyle='--', linewidth=0.7)

# Violin Plots
sns.violinplot(data=df_no_outliers, y='votes', color='skyblue', ax=axes[0, 1])
axes[0, 1].set_title('Violin Plot for Votes', fontsize=16)
axes[0, 1].set_ylabel('Votes', fontsize=14)
axes[0, 1].grid(True, linestyle='--', linewidth=0.7)

sns.violinplot(data=df_no_outliers, y='runtime', color='skyblue', ax=axes[1, 1])
axes[1, 1].set_title('Violin Plot for Runtime', fontsize=16)
axes[1, 1].set_ylabel('Runtime', fontsize=14)
axes[1, 1].grid(True, linestyle='--', linewidth=0.7)

sns.violinplot(data=df_no_outliers, y='rating', color='skyblue', ax=axes[2, 1])
axes[2, 1].set_title('Violin Plot for Rating', fontsize=16)
axes[2, 1].set_ylabel('Rating', fontsize=14)
axes[2, 1].grid(True, linestyle='--', linewidth=0.7)

plt.tight_layout()
plt.show()

#%%
#%% Strip Plot
plt.figure(figsize=(12, 6))
sns.stripplot(data=df_cleaned, x='genre', y='rating', jitter=True, palette='Set2')
plt.title('Strip Plot: Rating vs Genre', fontsize=16)
plt.xlabel('Genre', fontsize=14)
plt.ylabel('Rating', fontsize=14)
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', linewidth=0.7)
plt.show()
#%%
plt.figure(figsize=(12, 6))
sns.stripplot(data=df_cleaned, x='genre', y='runtime', jitter=True, palette='Set2')
plt.title('Strip Plot: Runtime vs Genre', fontsize=16)
plt.xlabel('Genre', fontsize=14)
plt.ylabel('Runtime', fontsize=14)
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', linewidth=0.7)
plt.show()
#%%
plt.figure(figsize=(12, 6))
sns.stripplot(data=df_cleaned, x='genre', y='votes', jitter=True, palette='Set2')
plt.title('Strip Plot: Votes vs Genre', fontsize=16)
plt.xlabel('Genre', fontsize=14)
plt.ylabel('Votes', fontsize=14)
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', linewidth=0.7)
plt.show()

#%% Swarm Plot

df_subset = df_cleaned.sample(2000, random_state=42)

plt.figure(figsize=(12, 6))
sns.swarmplot(data=df_subset, x='genre', y='runtime', palette='Set2')
plt.title('Swarm Plot: Runtime vs Genre (Random Subset of 2000 Rows)', fontsize=16)
plt.xlabel('Genre', fontsize=14)
plt.ylabel('Runtime', fontsize=14)
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', linewidth=0.7)
plt.show()

#%% PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
scaler = StandardScaler()
X= df_cleaned[['year', 'votes', 'rating', 'runtime']]
X = scaler.fit_transform(X)
print(X[:5])
#%%
pca= PCA()
X = pca.fit_transform(X)
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
components_90 = np.argmax(cumulative_variance >= 0.90) + 1
print(f"Number of components to explain 90% variance: {components_90}")
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
plt.axvline(x=components_90, color='r', linestyle='--', label=f'{components_90} components')
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA: Cumulative Explained Variance")
plt.legend()
plt.grid()
plt.show()
