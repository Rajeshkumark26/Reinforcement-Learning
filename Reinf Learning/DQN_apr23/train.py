from agent.agent import Agent
from functions import *
import sys
import pandas as pd
# import cudf as pd
import os

#set GPU Device
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


main_df=pd.DataFrame()
empty_list=[]

if len(sys.argv) != 4:
    print( "Usage: python train.py [stock] [window] [episodes]")
    exit()

stock_name, window_size, episode_count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])

agent = Agent(window_size)
data = getStockDataVec(stock_name)

# l = (data.size) -1
l = len(data) -1

#l=300
batch_size = 32

for e in range(episode_count + 1):
    print( "Episode " + str(e) + "/" + str(episode_count))
    state = getState(data, 0, window_size + 1)

    total_profit = 0
    agent.inventory = []


    for t in range(l):
#         state=np.reshape(state,(state.shape[0],state.shape[1],1))
        action = agent.act(state)

        # sit
        next_state = getState(data, t + 1, window_size + 1)
        reward = 0

        if action == 1: # buy
            agent.inventory.append(data[t])
            print( "Buy: " + formatPrice(data[t])+"  index:"+str(t))

        elif action == 2 and len(agent.inventory) > 0: # sell
            bought_price = agent.inventory.pop(0)
            reward = max(data[t] - bought_price, 0)
            total_profit += data[t] - bought_price
            empty_list.append({'Buy':bought_price,'Sell':data[t],'Profit':data[t] - bought_price})
            print( "Sell: " + formatPrice(data[t]) + " | Profit: " + formatPrice(data[t] - bought_price)+"  index:"+str(t))

        done = True if t == l - 1 else False
        agent.memory.append((state, action, reward, next_state, done))
        state = next_state

        if done:
            df1 = pd.DataFrame(empty_list, columns=['Buy','Sell','Profit'])
            path='./output/episode'+str(e)+'.csv'
            df1.to_csv(path)
            main_df=main_df.append(df1)
            print( "--------------------------------")
            print( "Total Profit: " + formatPrice(total_profit))
            print( "--------------------------------")

        if len(agent.memory) > batch_size:
            agent.expReplay(batch_size)

    if e % 1 == 0:
        agent.model.save("models/model_ep" + str(e))

main_df.to_csv('./main_df.csv')