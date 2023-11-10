# Curriculum_RL_for_Driving

Contains code for the paper "Improving Environment Robustness of Deep Reinforcement Learning Approaches for Autonomous Racing Using Bayesian Optimization-based Curriculum Learning" (Rohan Banerjee, Prishita Ray, Mark Campbell) presented at the Learning Robot Super Autonomy Workshop, IROS 2023.  

Has Default, Manual Curriculum and Bayesian Optimization-based Curriculum training for PPO RL agents in the OpenAI Gym CarRacing environment, and for an obstacles-based variant environment of CarRacing.  

Also contains top-down imagery code for both the environments under different turnrates and obstacle probability settings. 

Source code for obstacles variant environment: https://github.com/rohanb2018/carracing_obstacles     
Source code for top-down imagery: https://github.com/rohanb2018/carracing_fullmap   

The paper can be accessed here [add link]

<img width="800" height="600" src="https://github.com/PRISHIta123/Curriculum_RL_for_Driving/blob/main/kappa_p_combined.png?raw=true"/>

### Installation 

To install the required packages clone the repository and use the following:    
```pip install -r requirements.txt```  

### Guidelines 

1.Three methods of either default, manual curriculum or BO Curriculum can be used for training the PPO RL autonomous driving agent.    
2.Each of these methods can be used to train and evaluate the PPO agent in either constant/varying turnrates, obstacle probabilities or a combination of both.  

### Training/Testing PPO Agents with existing curricula 

To train a PPO Agent in a particular MODE (using previously found manual/BO curricula which are hardcoded) run:  
```python CarRacing_PPO.py --mode MODE --train True```  
where MODE is based on a particular training method/environment setting combination:  
1.PPO- Default turnrates  
2.PPO_Obstacles- Default obstacles/both  
3.PPO_Curriculum- Manual turnrates only curriculum  
4.PPO_Obstacles_Curriculum- Manual obstacles only curriculum  
5.PPO_Both_Curriculum- Manual both curriculum  
6.PPO_Curriculum_BO- BO turnrates only curriculum  
7.PPO_Obstacles_Curriculum_BO- BO obstacles only curriculum  
8.PPO_Both_Curriculum_BO- BO both curriculum  

To test a trained PPO Agent in a particular setting (after training) run:  
```python CarRacing_PPO.py --mode MODE --train False```  
where MODE follows the same format as above. 

### Searching for Curricula using Bayesian Optimization  

To run Bayesian Optimization to search for curricula in a particular MODE on your own run:  
```./BayesOpt.sh```  

When prompted to enter a MODE choose from the following:  
1.PPO_Curriculum_BO- BO turnrates only curriculum  
2.PPO_Obstacles_Curriculum_BO- BO obstacles only curriculum  
3.PPO_Both_Curriculum_BO- BO both curriculum  

To train a PPO Agent in a MODE (using new BO curricula) run the training code using:  
```python CarRacing_PPO.py --mode MODE --train True --ranges t1 t2 t3```  
where t1,t2,t3 are the 3 turnpoints output in the best curriculum found by Bayesian Optimization in the list format [t1,t2,t3].  

To observe the existing Bayesian Optimization runs/results see the Turnates_BO.pdf, Obstacles_BO.pdf and Both_BO.pdf files in the BO_runs folder.

### Training/Evaluation Curves Visualization  

To visualize the training/evaluation curves for each of the trained PPO agents run:  
```tensorboard --logdir runs```  

The tags correspond to the respective runs.
  
### CarRacing Generated Tracks Fullmap Visualization  

Navigate to the carracing_fullmap folder: ```cd carracing_fullmap```  

To visualize tracks with varying turnrates run:  
```python fullmap_carracing.py --turnrate TURNRATE```  
where turnrate can vary between [0.31,0.71]  

To visualize tracks with varying turnrates and obstacle probabilities run:  
```python fullmap_carracing.py --turnrate TURNRATE --obs_prob OBS_PROB```  
where turnrate can vary between [0.31,0.71], and obstacle probability can vary between [0.05, 0.13]  

Example tracks can be observed in the png files in the carracing_fullmap folder:  
TurnRate_TURNRATE.png (Tracks with varing turnrates)  
Both_TURNRATE_OBSPROB.png (Tracks with varying turnrates/obstacle probabilities)  

### Results  

Performance results can be accessed at the "Results.pdf" and the "Performance Comparison across Levels.pdf" files in the home directory. 

### Citation 

If you find our work useful in your research, please consider citing us:
[Add bibtex]
