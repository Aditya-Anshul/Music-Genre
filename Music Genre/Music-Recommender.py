import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib
from sklearn import tree
from sklearn.model_selection import train_test_split

music_data = pd.read_csv('music.csv')

X = music_data.drop(columns=['genre'])
Y = music_data['genre']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
model = DecisionTreeClassifier()
model.fit(x_train, y_train)
predictions = model.predict(x_test)

tree.export_graphviz(model, out_file='music-recommender.dot',
                     feature_names=['age', 'gender'],
                     class_names=sorted(Y.unique()),
                     label='all',
                     rounded=True,
                     filled=True)

age = input("Enter the Age: ")
g = input("Enter the Gender (M/F): ")
gender = -1
if g == 'M':
    gender = 1
elif g == 'F':
    gender = 0
else:
    print("Wrong input in gender")

while gender == -1:
    g = input("Enter the Gender (M/F): ")
    gender = -1
    if g == 'M':
        gender = 1
    elif g == 'F':
        gender = 0
    else:
        print("Wrong input in gender")

model = joblib.load('music-recommender.joblib')
predictions = model.predict([(age, gender)])

song_recommender = input("Do you want some recommendation about songs you might like (Y/N): ")
if song_recommender == 'Y':
    songs_data = pd.read_csv('songs_new.csv')
    songs_comb = songs_data.loc[songs_data.genre == predictions[0]]
    songs_comb.sort_values('title', ascending=True)
    print(songs_comb)
elif song_recommender == 'N':
    pass
else:
    print("Wrong input. Enter in 'Y' or 'N.'")

while song_recommender != 'Y' and song_recommender != 'N':
    song_recommender = input("Do you want some recommendation about songs you might like (Y/N): ")
    if song_recommender == 'Y':
        songs_data = pd.read_csv('songs_new.csv')
        songs_comb = songs_data.loc[songs_data.genre == predictions[0]]
        songs_comb.sort_values('title', ascending=True)
        print(songs_comb)
    elif song_recommender == 'N':
        pass
    else:
        print("Wrong input. Enter in 'Y' or 'N.'")
