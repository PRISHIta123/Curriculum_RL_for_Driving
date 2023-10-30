import numpy as np
import os

from scipy.optimize import minimize

import sys

import warnings
warnings.filterwarnings("ignore", message="delta_grad == 0.0. Check if the approximated function is linear.")

import gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EventCallback, BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, sync_envs_normalization

from typing import Any, Callable, Dict, List, Optional, Union

import argparse
import pickle
MODE=1
base_dir=''
lr=None
env=None
eval_env=None
## Manual Warmup Trials- with spacing (experimental design)

class CustomCallback(EventCallback):
    """
    Callback for evaluating an agent.

    .. warning::

      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``eval_freq = max(eval_freq // n_envs, 1)``

    :param eval_env: The environment used for initialization
    :param callback_on_new_best: Callback to trigger
        when there is a new best model according to the ``mean_reward``
    :param callback_after_eval: Callback to trigger after every evaluation
    :param n_eval_episodes: The number of episodes to test the agent
    :param eval_freq: Evaluate the agent every ``eval_freq`` call of the callback.
    :param log_path: Path to a folder where the evaluations (``evaluations.npz``)
        will be saved. It will be updated at each evaluation.
    :param best_model_save_path: Path to a folder where the best model
        according to performance on the eval env will be saved.
    :param deterministic: Whether the evaluation should
        use a stochastic or deterministic actions.
    :param render: Whether to render or not the environment during evaluation
    :param verbose: Verbosity level: 0 for no output, 1 for indicating information about evaluation results
    :param warn: Passed to ``evaluate_policy`` (warns if ``eval_env`` has not been
        wrapped with a Monitor wrapper)
    """

    def __init__(
        self,
        eval_env: Union[gym.Env, VecEnv],
        callback_on_new_best: Optional[BaseCallback] = None,
        callback_after_eval: Optional[BaseCallback] = None,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: Optional[str] = None,
        best_model_save_path: Optional[str] = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        warn: bool = True,
    ):
        super().__init__(callback_after_eval, verbose=verbose)

        self.callback_on_new_best = callback_on_new_best
        if self.callback_on_new_best is not None:
            # Give access to the parent
            self.callback_on_new_best.parent = self

        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        self.mean_rewards=[]
        self.deterministic = deterministic
        self.render = render
        self.warn = warn

        # Convert to VecEnv for consistency
        if not isinstance(eval_env, VecEnv):
            eval_env = DummyVecEnv([lambda: eval_env])

        self.eval_env = eval_env
        self.best_model_save_path = best_model_save_path
        # Logs will be written in ``evaluations.npz``
        if log_path is not None:
            log_path = os.path.join(log_path, "evaluations")
        self.log_path = log_path
        self.evaluations_results = []
        self.evaluations_timesteps = []
        self.evaluations_length = []
        # For computing success rate
        self._is_success_buffer = []
        self.evaluations_successes = []

    def _init_callback(self) -> None:
        # Does not work in some corner cases, where the wrapper is not the same
        if not isinstance(self.training_env, type(self.eval_env)):
            warnings.warn("Training and eval env are not of the same type" f"{self.training_env} != {self.eval_env}")

        # Create folders if needed
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)
        if self.log_path is not None:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

        # Init callback called on new best model
        if self.callback_on_new_best is not None:
            self.callback_on_new_best.init_callback(self.model)

    def _log_success_callback(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        """
        Callback passed to the  ``evaluate_policy`` function
        in order to log the success rate (when applicable),
        for instance when using HER.

        :param locals_:
        :param globals_:
        """
        info = locals_["info"]

        if locals_["done"]:
            maybe_is_success = info.get("is_success")
            if maybe_is_success is not None:
                self._is_success_buffer.append(maybe_is_success)

    def _on_step(self) -> bool:
        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e

            # Reset success rate buffer
            self._is_success_buffer = []

            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = mean_reward

            if self.verbose >= 1:
                print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose >= 1:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            self.mean_rewards.append(mean_reward)

            if mean_reward > self.best_mean_reward:
                if self.verbose >= 1:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = mean_reward
                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training

    def update_child_locals(self, locals_: Dict[str, Any]) -> None:
        """
        Update the references to the local variables.

        :param locals_: the local variables during rollout collection
        """
        if self.callback:
            self.callback.update_locals(locals_)

def fn(ranges):
    if MODE == 1:
        from gym.wrappers.time_limit import TimeLimit
        from TurnRates.car_racing_curriculum import CarRacingCurriculum

        env = TimeLimit(CarRacingCurriculum(ranges), max_episode_steps=1000)

        from gym.wrappers.time_limit import TimeLimit
        from TurnRates.car_racing_eval import CarRacingEval

        eval_env = TimeLimit(CarRacingEval(), max_episode_steps=1000)

        lr = 0.00025

    elif MODE == 2:
        from gym.wrappers.time_limit import TimeLimit
        from Obstacles.car_racing_obstacles_curriculum import CarRacingObstaclesCurriculum

        env = TimeLimit(CarRacingObstaclesCurriculum(ranges), max_episode_steps=1000)

        from gym.wrappers.time_limit import TimeLimit
        from Obstacles.car_racing_obstacles_eval import CarRacingObstaclesEval

        eval_env = TimeLimit(CarRacingObstaclesEval(), max_episode_steps=1000)

        lr = 0.000475

    elif MODE == 3:
        from gym.wrappers.time_limit import TimeLimit
        from Both.car_racing_obstacles_curriculum_both import CarRacingObstaclesCurriculumBoth

        env = TimeLimit(CarRacingObstaclesCurriculumBoth(ranges), max_episode_steps=1000)

        from gym.wrappers.time_limit import TimeLimit
        from Both.car_racing_obstacles_eval_both import CarRacingObstaclesEvalBoth

        eval_env = TimeLimit(CarRacingObstaclesEvalBoth(), max_episode_steps=1000)

        lr = 0.0002

    model = PPO('CnnPolicy', env=env, learning_rate=lr, n_steps=1000, batch_size=1000, verbose=0, seed=0)

    eval_callback = CustomCallback(eval_env=eval_env,
                                     n_eval_episodes=10,
                                     eval_freq=50000, verbose=1,
                                     deterministic=True, render=False)

    model.learn(total_timesteps=1000000, callback=eval_callback)

    model.env.close()
    env.close()
    eval_env.close()

    last_3_rewards_max= max(eval_callback.mean_rewards[-1],eval_callback.mean_rewards[-2],eval_callback.mean_rewards[-3])

    return last_3_rewards_max


def _get_neg_upper_confidence_bound(x_new, gauss_pr):

    # Using estimate from Gaussian surrogate instead of actual function for 
    # a new trial data point to avoid cost 

    mean_y_new, sigma_y_new = gauss_pr.predict(np.array([x_new]), return_std=True)

    # Exploration factor kappa:
    # 2.0- PPO Curriculum
    # 3.1- PPO Obstacles Curriculum
    # 1.9- PPO Both Curriculum

    if MODE==1:
        kappa=2.0

    elif MODE==2:
        kappa=3.1

    elif MODE==3:
        kappa=1.9

    neg_ucb= -1*mean_y_new - kappa*sigma_y_new
    
    return neg_ucb


def _acquisition_function(x, gauss_pr):
    return _get_neg_upper_confidence_bound(x, gauss_pr)

    
def _get_next_probable_point(x_init, gauss_pr):
     
    min_acq = float(sys.maxsize)
    x_optimal = None 
    
    # Trial with an array of random data points

    batch_size=30

    x_probs=[]

    for i in range(batch_size):

        if MODE==1:

            # Turnrates BO curriculum search space
            range0=np.random.uniform(150,250)
            range1=np.random.uniform(360,460)
            range2=np.random.uniform(660,760)

        elif MODE==2:

            # Obstacles BO curriculum search space
            range0=np.random.uniform(100,150)
            range1=np.random.uniform(220,260)
            range2=np.random.uniform(630,700)

        elif MODE==3:

            # Both BO curriculum search space
            range0=np.random.uniform(150,250)
            range1=np.random.uniform(330,450)
            range2=np.random.uniform(730,830)

        x=np.array([range0,range1,range2])
        x_probs.append(x)

    x_probs=np.array(x_probs)
    
    for x_start in x_probs:

        response = minimize(fun=_acquisition_function, x0=x_start, args=(gauss_pr,), method='BFGS')

        print("X:",x_start,"Y:",response.fun)

        if response.fun < min_acq:
            min_acq= response.fun
            x_optimal = x_start
    
    return x_optimal, min_acq

if __name__ == "__main__":

    parser = argparse.ArgumentParser("BayesOpt")
    parser.add_argument("mode", type=int,
                        help="Bayesian Optimization Mode: 1.BO Turnrates curriculum \n 2.BO Obstacle probability curriculum \n 3.BO Both curriculum")
    parser.add_argument("ITER", help="iteration number to unpickle values", type=int)
    args = parser.parse_args()
    ITER= args.ITER

    MAX_ITER=5

    if args.mode==1 or args.mode==3:
        MAX_ITER=18

    elif args.mode==2:
        MAX_ITER=22

    MODE=args.mode

    #Unpickle values

    if MODE==1:
        base_dir='./Turnrates_BO/vars/'
    elif MODE==2:
        base_dir='./Obstacles_BO/vars/'
    elif MODE==3:
        base_dir='./Both_BO/vars/'

    with open(base_dir+'x_init'+str(ITER-1)+'.pkl','rb') as f:
        x_init = pickle.load(f)

    with open(base_dir+'y_init'+str(ITER-1)+'.pkl','rb') as f:
        y_init = pickle.load(f)
            
    with open(base_dir+'gauss_pr'+str(ITER-1)+'.pkl','rb') as f:
        gauss_pr = pickle.load(f)

    with open(base_dir+'y_max_ind'+str(ITER-1)+'.pkl','rb') as f:
        y_max_ind = pickle.load(f)

    with open(base_dir+'y_max'+str(ITER-1)+'.pkl','rb') as f:
        y_max = pickle.load(f)
        
    with open(base_dir+'optimal_x'+str(ITER-1)+'.pkl','rb') as f:
        optimal_x = pickle.load(f)

    with open(base_dir+'optimal_acq'+str(ITER-1)+'.pkl','rb') as f:
        optimal_acq= pickle.load(f)

    with open(base_dir+'distances_'+str(ITER-1)+'.pkl','rb') as f:
        distances_ = pickle.load(f)

    with open(base_dir+'best_samples_'+str(ITER-1)+'.pkl','rb') as f:
        best_samples_ = pickle.load(f)

    if ITER>5:
        with open(base_dir+'prev_x'+str(ITER-1)+'.pkl','rb') as f:
            prev_x = pickle.load(f)

    #BayesOpt trials
    if ITER<=MAX_ITER:

        print("Trial: ",ITER)
        
        gauss_pr.fit(x_init, y_init)
        
        x_next, acq = _get_next_probable_point(x_init, gauss_pr)

        range0=x_next[0]
        range1=x_next[1]
        range2=x_next[2]

        ranges=np.array([range0,range1,range2])

        print("Ranges:",ranges)

        y_next= fn(ranges)

        print("Ranges:",ranges)
        print("Reward:",y_next)
        print("Neg_UCB:",acq)

        x_next_app= np.expand_dims(x_next, axis=0)
        y_next_app= np.array(y_next)
        
        
        x_init = np.r_[x_init, x_next_app]
        y_init = np.r_[y_init, y_next_app]
        
        if y_next > y_max:
            y_max = y_next
            optimal_x = x_next
            optimal_acq = acq

        if ITER == 5:
             prev_x = x_next

        else:
             distances_.append(np.linalg.norm(prev_x - x_next))
             prev_x = x_next
        
        best_samples_ = best_samples_.append({"y": y_max, "acq": optimal_acq},ignore_index=True)

        #Pickle x_init, y_init, gauss_pr and everything above

        with open(base_dir+'x_init'+str(ITER)+'.pkl','wb') as f:
            pickle.dump(x_init, f)

        with open(base_dir+'y_init'+str(ITER)+'.pkl','wb') as f:
            pickle.dump(y_init, f)
            
        with open(base_dir+'gauss_pr'+str(ITER)+'.pkl','wb') as f:
            pickle.dump(gauss_pr, f)

        with open(base_dir+'y_max_ind'+str(ITER)+'.pkl','wb') as f:
            pickle.dump(y_max_ind, f)

        with open(base_dir+'y_max'+str(ITER)+'.pkl','wb') as f:
            pickle.dump(y_max, f)
        
        with open(base_dir+'optimal_x'+str(ITER)+'.pkl','wb') as f:
            pickle.dump(optimal_x, f)

        with open(base_dir+'optimal_acq'+str(ITER)+'.pkl','wb') as f:
            pickle.dump(optimal_acq, f)

        with open(base_dir+'distances_'+str(ITER)+'.pkl','wb') as f:
            pickle.dump(distances_, f)

        with open(base_dir+'best_samples_'+str(ITER)+'.pkl','wb') as f:
            pickle.dump(best_samples_, f)

        with open(base_dir+'prev_x'+str(ITER)+'.pkl','wb') as f:
            pickle.dump(prev_x, f)

        if ITER == MAX_ITER:
            print(optimal_x, y_max, optimal_acq)



    

        

    

