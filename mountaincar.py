import gym
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy as np
from progress.bar import IncrementalBar

env = gym.make("MountainCar-v0")

def randomGames():
    global env
    training_data = []
    accepted_scores = []

    bar = IncrementalBar("Games",max=10000)
    for games in range(10000):
        score = 0
        game_memory = []
        prev_observation = []
        observation = env.reset()
        for step in range(200):
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if len(prev_observation) > 0:
                game_memory.append([prev_observation, action])
            prev_observation = observation
            if observation[0] > -0.2:
                reward = 1
            score += reward
            if done:
                bar.next()
                break

        if score > -197:
            accepted_scores.append(score)
            for data in game_memory:
                if data[1] == 1:
                    output = [0, 1, 0]
                elif data[1] == 0:
                    output = [1, 0, 0]
                elif data[1] == 2:
                    output = [0, 0, 1]
                training_data.append([data[0], output])
    bar.finish()

    return training_data

def makeAndFitModel():
    global training_data
    X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]))
    Y = np.array([i[1] for i in training_data]).reshape(-1, len(training_data[0][1]))

    model = Sequential()
    model.add(Dense(200, input_dim=2, activation="relu"))
    model.add(Dense(50, activation="relu"))
    model.add(Dense(50, activation="relu"))
    model.add(Dense(50, activation="relu"))
    model.add(Dense(3, activation="sigmoid"))

    model.compile(loss="mean_squared_error",optimizer="adam")

    model.fit(X,Y,epochs=100)

    return model

def loadModel():
    with open("mountainmodel.json","r") as file:
        model_json = file.read()
        loaded_model = model_from_json(model_json)
    loaded_model.load_weights("mountainmodel.h5")

    return loaded_model

def actually_play():
    trained_model = loadModel()
    scores = []
    choices = []
    for each_game in range(50):
        print(each_game)
        env.reset()
        score = 0
        game_memory = []
        prev_obs = []
        for step_index in range(200):
            print("      "+ str(step_index))
            env.render()
            if len(prev_obs)==0:
                action = env.action_space.sample()
            else:
                action = np.argmax(trained_model.predict(prev_obs.reshape(-1, len(prev_obs)))[0])
            
            choices.append(action)
            new_observation, reward, done, info = env.step(action)
            prev_obs = new_observation
            game_memory.append([new_observation, action])
            score += reward
            if done:
                break

        scores.append(score)

    print(scores)
    print('Average Score:',sum(scores)/len(scores))
    print('choice 1:{}  choice 0:{} choice 2:{}'.format(choices.count(1)/len(choices),choices.count(0)/len(choices),choices.count(2)/len(choices)))

env.close()