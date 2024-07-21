import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import plotly.express as px
import scipy.stats as stats
from sklearn.model_selection import cross_val_score
df=pd.read_csv('IMDb Movies India.csv',encoding='latin1')
df.head(10)
df.info()
df.columns
df.describe()
print("unique count")
print(df.nunique())
print("Null count")
df.isnull().sum()
def missing_values_percent(dataframe):
    missing_values = dataframe.isna().sum()
    percentage_missing = (missing_values / len(dataframe) * 100).round(2)

    result_movie = pd.DataFrame({'Missing Values': missing_values, 'Percentage': percentage_missing})
    result_movie['Percentage'] = result_movie['Percentage'].astype(str) + '%'

    return result_movie


result = missing_values_percent(df)
result
sns.heatmap(df.isnull())
missing_values = df.isnull().sum()

# Convert the Series to a DataFrame for plotting
missing_values_df = missing_values.reset_index()
missing_values_df.columns = ['Column', 'Missing Values']

# Plot using seaborn's barplot
plt.figure(figsize=(5, 4))
sns.barplot(x='Column', y='Missing Values', data=missing_values_df)
plt.xticks(rotation=90)  # Rotate column names for better visibility
plt.title('Number of Missing Values in Each Column')
plt.show()
df['Rating'].fillna(df['Rating'].mean())
df=df.dropna(subset=['Year'],axis=0)
df['Year'].head()
df['Year'] = df['Year'].str.replace('(', '', regex=False).str.replace(')', '', regex=False).astype(int)

df.head()
result = missing_values_percent(df)
result
df['Duration'] = df['Duration'].fillna('0').astype(str)

    # Remove ' min' from 'Duration'
df['Duration'] = df['Duration'].str.replace(' min', '')

    # Convert 'Duration' to integers
df['Duration'] = df['Duration'].astype(int)
print(max(df['Duration']))
print(min(df['Duration']))
print(np.mean(df['Duration']))
print(max(df['Duration']))

df['Duration']=df['Duration'].astype(int)
sns.boxplot(x=df['Duration'])
# plt.figure(figsize=(106, 10))
sns.jointplot(data=df, y='Genre', x='Duration',height=69,dropna=True,kind='hist')
plt.xticks(rotation=45)
plt.show()
median_duration_by_genre = df.groupby('Genre')['Duration'].median()

# Display median duration by genre
print("\nMedian duration by genre:")
print(median_duration_by_genre)

# Debug: Print rows with Duration == 0 before replacement
print("\nRows with Duration == 0 before replacement:")
print(df[df['Duration'] == 0])

# Replace Duration == 0 with median duration by genre inplace
for genre, median_duration in median_duration_by_genre.items():
    df.loc[(df['Duration'] == 0) & (df['Genre'] == genre), 'Duration'] = median_duration
    # Debug: Print rows with Duration == 0 after replacement
print("\nRows with Duration == 0 after replacement:")
print(df['Duration'].value_counts())
median_duration_by_director = df.groupby('Director')['Duration'].median()

# Replace Duration == 0 with median duration by Director inplace
for director, median_duration in median_duration_by_director.items():
    df.loc[(df['Duration'] == 0) & (df['Director'] == director), 'Duration'] = median_duration

print("\nCount of each Duration value:")
print(df['Duration'].value_counts())

actors=['Actor 1','Actor 2',  'Actor 3']
for actor in actors:
    median_duration_by_actor = df.groupby(actor)['Duration'].median()

    for act, median_duration in median_duration_by_actor.items():
        df.loc[(df['Duration'] == 0) & (df[actor] == act), 'Duration'] = median_duration
    print(f"\nCount of each Duration value: when grouping by {actor}")
    print(df['Duration'].value_counts())
    df['Duration']=df['Duration'].astype(int)
    sns.boxplot(x=df['Duration'])
    print(max(df['Duration']))
    df=df.dropna(subset=['Duration'],axis=0)
    df=df[df['Duration'] >= 60]
    df.head()
    sns.displot(df['Duration'])
    df.info()
    sns.displot(df['Rating'])
    df=df.dropna(subset=['Rating'],axis=0)
    result = missing_values_percent(df)
    result
    df["Director"]=df["Director"].fillna('Dilip Bose')
    
    result = missing_values_percent(df)
    result
    df['Genre'] = df['Genre'].str.split(', ')
    df = df.explode('Genre')
    df['Genre'].fillna(df['Genre'].mode()[0], inplace=True)
    df.head()

    df=df.dropna(subset=['Actor 1','Actor 2','Actor 3'],axis=0)
    result = missing_values_percent(df)
    result
    df.info()
    
    df['Votes'] = df['Votes'].astype(str)

    df['Votes'] = df['Votes'].str.replace(',', '', regex=False).astype(int)

    df.head(10)
    fig_year = px.histogram(df, x='Year', histnorm='probability density', nbins=30)
    fig_year.update_traces(selector=dict(type='histogram'))
    fig_year.update_layout(
        title='Distribution of Year',
        title_x=0.5,
        title_pad=dict(t=20),
        title_font=dict(size=20),
        xaxis_title='Year',
        yaxis_title='Probability Density',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        bargap=0.02,
        plot_bgcolor='white'
    )
    fig_dur = px.histogram(df, x = 'Duration', histnorm='probability density', nbins = 40)
    fig_dur.update_traces(selector=dict(type='histogram'))
    fig_dur.update_layout(
        title='Distribution of Duration', 
        title_x=0.5, title_pad=dict(t=20),
        title_font=dict(size=20), xaxis_title='Duration',
        yaxis_title='Probability Density',
        xaxis=dict(showgrid=False), 
        yaxis=dict(showgrid=False),
        bargap=0.02, 
        plot_bgcolor = 'white')
    fig_dur.show()
    fig_rat = px.histogram(df, x = 'Rating', histnorm='probability density', nbins = 40)
    fig_rat.update_traces(selector=dict(type='histogram'))
    fig_rat.update_layout(title='Distribution of Rating', 
                          title_x=0.5, 
                          title_pad=dict(t=20),
                          title_font=dict(size=20),
                          xaxis_title='Rating',
                          yaxis_title='Probability Density',
                          xaxis=dict(showgrid=False),
                          yaxis=dict(showgrid=False),
                          bargap=0.02,
                          plot_bgcolor = 'white')
    fig_rat.show()
    fig_vot = px.box(df, x = 'Votes')
    fig_vot.update_layout(title='Distribution of Votes', 
                            title_x=0.5,
                            title_pad=dict(t=20), 
                            title_font=dict(size=20),
                            xaxis_title='Votes',
                            yaxis_title='Probability Density', 
                            xaxis=dict(showgrid=False), 
                            yaxis=dict(showgrid=False), 
                            plot_bgcolor = 'white')
    fig_vot.show()
    rel_dur_rat = px.scatter(df, x = 'Duration', y = 'Rating', color = "Rating")
    rel_dur_rat.update_layout(title='Rating v/s Duration of Movie',
                              title_x=0.5,
                              title_pad=dict(t=20),
                              title_font=dict(size=20),
                              xaxis_title='Duration of Movie in Minutes',
                              yaxis_title='Rating of a movie',
                              xaxis=dict(showgrid=False), 
                              yaxis=dict(showgrid=False),
                              plot_bgcolor = 'white')
    rel_dur_rat.show()
    rel_dur_rat = px.scatter(df, x = 'Actor 1', y = 'Rating', color = "Rating")
    rel_dur_rat.update_layout(title='Rating v/s Actor 1',
                              title_x=0.5,
                              title_pad=dict(t=20),
                              title_font=dict(size=20),
                              xaxis_title='Actor 1',
                              yaxis_title='Rating of a movie',
                              xaxis=dict(showgrid=False), 
                              yaxis=dict(showgrid=False),
                              plot_bgcolor = 'white')
    rel_dur_rat.show()
    rel_dur_rat = px.scatter(df, x = 'Actor 2', y = 'Rating', color = "Rating")
    rel_dur_rat.update_layout(title='Rating v/s Actor 2',
                              title_x=0.5,
                              title_pad=dict(t=20),
                              title_font=dict(size=20),
                              xaxis_title='Actor 2',
                              yaxis_title='Rating of a movie',
                              xaxis=dict(showgrid=False), 
                              yaxis=dict(showgrid=False),
                              plot_bgcolor = 'white')
    rel_dur_rat.show()
    rel_dur_rat = px.scatter(df, x = 'Actor 3', y = 'Rating', color = "Rating")
    rel_dur_rat.update_layout(title='Rating v/s Actor 3',
                              title_x=0.5,
                              title_pad=dict(t=20),
                              title_font=dict(size=20),
                              xaxis_title='Actor 3',
                              yaxis_title='Rating of a movie',
                              xaxis=dict(showgrid=False), 
                              yaxis=dict(showgrid=False),
                              plot_bgcolor = 'white')
    rel_dur_rat.show()
    fig_rat_votes = px.scatter(df, x = 'Rating', y = 'Votes', color = "Votes")
    fig_rat_votes.update_layout(title='Getting Look at  Ratings impact on Votes ',
                                title_x=0.5, 
                                title_pad=dict(t=20), 
                                title_font=dict(size=20), 
                                xaxis_title='Ratings of Movies', 
                                yaxis_title='Votes of movies', 
                                xaxis=dict(showgrid=False), 
                                yaxis=dict(showgrid=False),
                                plot_bgcolor = 'white')
    fig_rat_votes.show()
    
    if 'Name' in df.columns:
     df.drop('Name', axis=1, inplace=True)
    
    else:
     print("'Name' column does not exist in DataFrame.")
    
    g_mean_rat = df.groupby('Genre')['Rating'].transform('mean')
    df['G_mean_rat'] = g_mean_rat
    
    dir_mean_rat = df.groupby('Director')['Rating'].transform('mean')
    df['Dir_enc'] = dir_mean_rat
    
    a1_mean_rat = df.groupby('Actor 1')['Rating'].transform('mean')
    df['A1_enc'] = a1_mean_rat
    
    a2_mean_rat = df.groupby('Actor 2')['Rating'].transform('mean')
    df['A2_enc'] = a2_mean_rat
    
    a3_mean_rat = df.groupby('Actor 3')['Rating'].transform('mean')
    df['A3_enc'] = a3_mean_rat
    df.head(10)
    X = df[['Year', 'Votes', 'Duration', 'G_mean_rat', 'Dir_enc', 'A1_enc', 'A2_enc', 'A3_enc']]
    y = df['Rating']
    print(X.shape)
    print(y.shape)
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=2)
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)
    X_train.info()
    
    lr = LinearRegression()
    lr.fit(X_train,y_train)
    lr_pred = lr.predict(X_test)
    print('The performance evaluation of Linear Regression is below:')
    print('Mean squared error:', metrics.mean_squared_error(y_test, lr_pred))
    print('Mean absolute error:', metrics.mean_absolute_error(y_test, lr_pred))
    print('R2 score:', metrics.r2_score(y_test, lr_pred))
    print('\n', '='*100, '\n')
    
    # Perform 5-fold cross-validation for Linear Regression
    cv_scores_lr = cross_val_score(lr, X, y, cv=8, scoring='r2')
    print('Linear Regression 5-fold cross-validation R2 scores:', cv_scores_lr)
    print('Mean R2 score:', cv_scores_lr.mean())
    
dt_regressor = DecisionTreeRegressor(random_state=2)
dt_regressor.fit(X_train, y_train)
y_pred = dt_regressor.predict(X_test)
print('The performance evaluation of Decision Tree Regressor is below:')
print('Mean squared error:', metrics.mean_squared_error(y_test, y_pred))
print('Mean absolute error:', metrics.mean_absolute_error(y_test, y_pred))
print('R2 score:', metrics.r2_score(y_test, y_pred))

# Perform 5-fold cross-validation for Decision Tree Regressor
cv_scores_dt = cross_val_score(dt_regressor, X, y, cv=8, scoring='r2')
print('\nDecision Tree Regressor 5-fold cross-validation R2 scores:', cv_scores_dt)
print('Mean R2 score:', cv_scores_dt.mean())

rf = RandomForestRegressor()
rf.fit(X_train,y_train)
rf_pred = rf.predict(X_test)
print('The performance evaluation of Random Forest Regressor is below:')
print('Mean squared error:', metrics.mean_squared_error(y_test, rf_pred))
print('Mean absolute error:', metrics.mean_absolute_error(y_test, rf_pred))
print('R2 score:', metrics.r2_score(y_test, rf_pred))

# Perform 5-fold cross-validation for Random Forest Regressor
cv_scores_rf = cross_val_score(rf, X, y, cv=8, scoring='r2')
print('\nRandom Forest Regressor 5-fold cross-validation R2 scores:', cv_scores_rf)
print('Mean R2 score:', cv_scores_rf.mean())
Featured_df = df.copy()
Featured_df.to_csv('Featured_df.csv', index=False)

sns.displot(df['Votes'])
plt.xlim(1, 1000)  
plt.ylim(0, 10000) 
plt.show()
sns.displot(df['Year'])
xol=[ 'Duration', 'G_mean_rat', 'Dir_enc', 'A1_enc', 'A2_enc', 'A3_enc']
for x in xol:
    sns.displot(df[x])
    scaler=StandardScaler()
    x_scaled=scaler.fit_transform(df[['Duration', 'G_mean_rat', 'Dir_enc', 'A1_enc', 'A2_enc', 'A3_enc']])
    std_df=pd.DataFrame(columns=[ 'Duration', 'G_mean_rat', 'Dir_enc', 'A1_enc', 'A2_enc', 'A3_enc'],data=x_scaled)
    minmax=MinMaxScaler()
    x_minmax=minmax.fit_transform(df[['Year','Votes']])
    minmax_df = pd.DataFrame(data=x_minmax, columns=['Year', 'Votes'])
    scaled_df = pd.concat([std_df, minmax_df], axis=1)
    scaled_df.head()
    xol=['Year','Votes', 'Duration', 'G_mean_rat', 'Dir_enc', 'A1_enc', 'A2_enc', 'A3_enc']
    for x in xol:
        sns.displot(scaled_df[x])
        X = df[['Year', 'Votes', 'Duration', 'G_mean_rat', 'Dir_enc', 'A1_enc', 'A2_enc', 'A3_enc']]
        y=df['Rating']
        
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=2)
        print(X_train.shape)
        print(y_train.shape)
        print(X_test.shape)
        print(y_test.shape)
        X_train.info()
        lr = LinearRegression()
lr.fit(X_train,y_train)
lr_pred = lr.predict(X_test)
print('The performance evaluation of Linear Regression is below:')
print('Mean squared error:', metrics.mean_squared_error(y_test, lr_pred))
print('Mean absolute error:', metrics.mean_absolute_error(y_test, lr_pred))
print('R2 score:', metrics.r2_score(y_test, lr_pred))
print('\n', '='*100, '\n')

# Perform 5-fold cross-validation for Linear Regression
cv_scores_lr = cross_val_score(lr, X, y, cv=8, scoring='r2')
print('Linear Regression 5-fold cross-validation R2 scores:', cv_scores_lr)
print('Mean R2 score:', cv_scores_lr.mean())

dt_regressor = DecisionTreeRegressor(random_state=2)
dt_regressor.fit(X_train, y_train)
y_pred = dt_regressor.predict(X_test)
print('The performance evaluation of Decision Tree Regressor is below: ', '\n')
print('Mean squared error: ', metrics.mean_squared_error(y_test, y_pred))
print('Mean absolute error: ', metrics.mean_absolute_error(y_test, y_pred))
print('R2 score: ', metrics.r2_score(y_test, y_pred))
cv_scores_dt = cross_val_score(dt_regressor, X, y, cv=8, scoring='r2')
print('\nDecision Tree Regressor 5-fold cross-validation R2 scores:', cv_scores_dt)
print('Mean R2 score:', cv_scores_dt.mean())

rf = RandomForestRegressor()
rf.fit(X_train,y_train)
rf_pred = rf.predict(X_test)
print('The performance evaluation of Random Forest Regressor is below: ', '\n')
print('Mean squared error: ',metrics.mean_squared_error(y_test, rf_pred))
print('Mean absolute error: ',metrics.mean_absolute_error(y_test, rf_pred))
print('R2 score: ',metrics.r2_score(y_test, rf_pred))
cv_scores_rf = cross_val_score(rf, X, y, cv=8, scoring='r2')
print('\nRandom Forest Regressor 5-fold cross-validation R2 scores:', cv_scores_rf)
print('Mean R2 score:', cv_scores_rf.mean())
