import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score



# this is the simple way to start with ml
n = int(input('how many predications do you want to make :- '))
for i in range(n):
    music_data = pd.read_csv('music.csv')
    x = music_data.drop(columns=['fruit'])
    y = music_data['fruit']
    print(music_data)

    model = DecisionTreeClassifier()
    model.fit(x, y)
    ag = int(input('enter the age of the person :- '))
    gender = int(input('enter the gender of the person :- '))
    # array = [ag,gende]
    predications = model.predict([[ag, gender]])
    print(predications)


# now using the train and test function
music_data = pd.read_csv('music.csv')
x = music_data.drop(columns=['fruit'])
y = music_data['fruit']
x_train,y_train,x_test,y_test = train_test_split(x,y,test_size=0.3)

model = DecisionTreeClassifier()
model.fit(x_train,y_train)
predications = model.predict(x_test)
print(predications)

score = accuracy_score(y_test,predications)
print(score)
