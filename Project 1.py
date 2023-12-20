#!/usr/bin/env python
# coding: utf-8

# In[12]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


xG=pd.read_csv('understat.com.csv')
xG_pg=pd.read_csv('understat_per_game.csv')  
xG_player=pd.read_csv('Data.csv')  
results=pd.read_csv('Results.csv')

xG.rename(columns={xG.columns[1]: 'year'}, inplace=True)
xG.rename(columns={xG.columns[0]: 'league'}, inplace=True)


xG.to_csv('understat.com.csv', index=False)

xG=xG[xG['league'].str.strip() != 'RFPL']


# In[3]:





# In[10]:


pivot_table = xG.pivot_table(index='league', columns='year', values='xG', aggfunc='mean')
plt.figure(figsize=(10, 6))
sns.heatmap(pivot_table, cmap='coolwarm', annot=True)
plt.title('Heatmap of Average xG Values Across Leagues and Years')
plt.ylabel('League')
plt.xlabel('Year')

plt.show()


# In[18]:


xG.describe()


# In[20]:


leagues = ['Ligue_1', 'Serie_A', 'Bundesliga', 'EPL', 'La_liga']

for league in leagues:
    league_data = filtered_data[filtered_data['league'] == league]

    yearly_xG = league_data.groupby('year')['xG'].mean()

    plt.figure(figsize=(10, 6))
    plt.plot(yearly_xG, marker='o')
    plt.title(f'Average xG Over Years in {league}')
    plt.ylabel('Average xG')
    plt.xlabel('Year')
    plt.show()


# In[29]:


selected_years=[2018, 2019]
selected_leagues=['La_liga','EPL'] 
fig,axs=plt.subplots(len(selected_years),len(selected_leagues), figsize=(15, 10),constrained_layout=True)
for i, year in enumerate(selected_years):
    for j, league in enumerate(selected_leagues):
        filtered_data=xG[(xG['year']==year)&(xG['league'] ==league)]
        sorted_data=filtered_data.sort_values(by='position')
        ax1=axs[i,j]
        ax2=ax1.twinx()
        ax1.plot(sorted_data['position'],sorted_data['xG'],marker='o',color='blue', label='xG')
        ax1.set_ylabel('xG',color='blue')
        ax1.tick_params(axis='y',labelcolor='blue')
        ax2.plot(sorted_data['position'], sorted_data['position'],linestyle='--', marker='o',color='red',label='Position')
        ax2.set_ylabel('Position',color='red')
        ax2.tick_params(axis='y',labelcolor='red')
        ax1.set_title(f'{league}-{year}')
        ax1.set_xlabel('Teams(sorted by position)')

plt.show()


# In[9]:


plt.figure(figsize=(10, 6))
plt.scatter(xG_pg['xG'], xG_pg['scored'],alpha=0.5)
m,b= np.polyfit(xG_pg['xG'], xG_pg['scored'], 1)
plt.grid(True)
plt.title('Relationship between Expected Goals (xG) and Goals Scored')
plt.xlabel('Expected Goals (xG)')
plt.ylabel('Goals Scored')

plt.show()


# In[12]:


xG_player.describe()


# In[23]:


plt.hist(xG_player['Goals'], bins=20, label='Goals',color='b')
plt.hist(xG_player['xG'], bins=20, alpha=0.6, label='xG', color='y')

plt.xlabel('Goals / Expected Goals')
plt.ylabel('Frequency')
plt.title('Distribution of Goals and Expected Goals')
plt.legend(loc='upper right')
plt.show()


# In[15]:


plt.figure(figsize=(10,6))
plt.hist(xG_player['Goals'],bins=20,color='blue',edgecolor='black')
plt.title('Distribution of Goals Scored')
plt.xlabel('Goals Scored')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


# In[16]:


plt.figure(figsize=(10,6))
plt.scatter(xG_player['xG'],xG_player['Goals'],alpha=0.5)
plt.title('Goals vs. Expected Goals (xG)')
plt.xlabel('Expected Goals (xG)')
plt.ylabel('Goals Scored')
plt.grid(True)
plt.show()



# In[17]:


avg_xG_per_year=xG_player.groupby('Year')['xG'].mean().reset_index()

plt.figure(figsize=(10,6))
plt.plot(avg_xG_per_year['Year'], avg_xG_per_year['xG'], marker='o')
plt.title('Average Expected Goals (xG) Over the Years')
plt.xlabel('Year')
plt.ylabel('Average Expected Goals (xG)')
plt.grid(True)
plt.show()


# In[8]:


def determine_home_result(row):
    if row['home_score'] > row['away_score']:
        return 'win'
    elif row['home_score'] < row['away_score']:
        return 'loss'
    else:
        return 'draw'

results['home_result'] = results.apply(determine_home_result, axis=1)

def calculate_home_advantage(data):
    home_win_rate = round((data['home_result'] == 'win').mean() * 100, 2)
    home_loss_rate = round((data['home_result'] == 'loss').mean() * 100, 2)
    home_draw_rate = round((data['home_result'] == 'draw').mean() * 100, 2)
    return home_win_rate, home_loss_rate, home_draw_rate

non_neutral_matches = results[results['neutral'] == False]
non_neutral_advantage = calculate_home_advantage(non_neutral_matches)

neutral_matches = results[results['neutral'] == True]
neutral_advantage = calculate_home_advantage(neutral_matches)

print("Non-Neutral Ground Home Advantage:", non_neutral_advantage)
print("Neutral Ground Home Advantage:", neutral_advantage)


# In[ ]:





# In[ ]:




