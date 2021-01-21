# Portfolio project with Reinforcement Learning
## 1) Data Collection

refer [Open source](https://github.com/gyusu/Creon-Datareader "대신증권 데이터 수집 github")  
with **preprocessor.py** split data into 3:1 (train:test).


## 2) Momentum-based Investment Strategies  
I implemented these strategies based on this paper: [Online Portfolio Selection: A Survey](https://arxiv.org/pdf/1212.2129.pdf) 

- **BAH (Buy and Hold)**  
Equally invest m assets at once
  
- **Best**  
Choose the best profitable asset in a hindsight   

- **CRP (Constant Rebalanced Portfolio)**  
Rebalance assets to a fixed ratio every period  

- **EG (Exponential Gradient)**  
It is based on "Follow-the-Winner" approach  
It aims to maximize log-return with little change in portfolio value 
     
- **Anticor (Anti correlation)**  
It is based on "Follow-the-Looser" approach  
It assumes mean-reversion considering cross-correlation and auto-correlation
  
- **OLMAR (Online Moving Average Reversion)**  
It predicts future price with moving average  
This method minimizes the change of portfolio value which yields profit more than certain value (epsilon)  
<!-- **WMAMR (Weighted Moving Average Mean Reversion** -->


## 3) Simple RL Application

Coming soon !

## 4) Results
<p align='center'>
<img src="/imgs/sample_data.png" width="400"    />
<img src="/imgs/sample_cul.png" width="400"    />
<p/>
 
visualize.show_allocation_ratio function  

> Toy example

<p align='center'>
<img src="/imgs/sample_portfolio_ratio.png" width="400"    />
<p/>

