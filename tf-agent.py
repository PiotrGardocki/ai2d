import os
# Keep using keras-2 (tf-keras) rather than keras-3 (keras).
#os.environ['TF_USE_LEGACY_KERAS'] = '1'

import imageio
import matplotlib.pyplot as plt
import pyvirtualdisplay
import reverb

import tensorflow as tf
import keras

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import py_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.networks import sequential
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.specs import tensor_spec
from tf_agents.utils import common
from tf_agents.train import learner
from tf_agents.train import triggers

from env import SpaceInvaders2Env

# Set up a virtual display for rendering OpenAI gym environments.
display = pyvirtualdisplay.Display(visible=0, size=(720, 720)).start()

num_iterations = 100000

data_dir = "custom_game_data"

initial_collect_steps = 1000
collect_steps_per_iteration = 1
replay_buffer_max_length = 100000

batch_size = 64
learning_rate = 1e-3
log_interval = 1000

num_eval_episodes = 10
eval_interval = 4000
policy_save_interval = 5000

env_name = 'games/SpaceInvaders-v2'
def new_env():
    # return suite_gym.load(env_name, gym_kwargs={'render_mode': 'simple_array'})
    return suite_gym.load(env_name, gym_kwargs={'render_mode': 'rgb_array'})

env = new_env()
env.reset()

# print('Observation Spec:')
# print(env.time_step_spec().observation)

# print('Reward Spec:')
# print(env.time_step_spec().reward)

print('Action Spec:')
print(env.action_spec())

# time_step = env.reset()
# print('Time step:')
#print(time_step)

# action = np.array(1, dtype=np.int32)

# next_time_step = env.step(action)
# print('Next time step:')
#print(next_time_step)

train_py_env = new_env()
eval_py_env = new_env()

train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

action_tensor_spec = tensor_spec.from_spec(env.action_spec())
num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1

# q_net = keras.Sequential([
#     keras.layers.Flatten(),
#     keras.layers.Dense(100, 'relu', kernel_initializer=keras.initializers.VarianceScaling(
#         scale=2.0, mode='fan_in', distribution='truncated_normal')),
#     keras.layers.Dense(50, 'relu', kernel_initializer=keras.initializers.VarianceScaling(
#         scale=2.0, mode='fan_in', distribution='truncated_normal')),
#     keras.layers.Dense(num_actions, activation=None, kernel_initializer=keras.initializers.RandomUniform(
#         minval=-0.03, maxval=0.03), bias_initializer=keras.initializers.Constant(-0.2)),
# ])

# q_net = sequential.Sequential([
#     keras.layers.Flatten(),
#     keras.layers.Dense(100, 'relu'),
#     keras.layers.Dense(100, 'relu'),
#     keras.layers.Dense(50, 'relu'),
#     keras.layers.Dense(num_actions),
# ])

q_net = sequential.Sequential([
    #tf.keras.layers.Input((720, 720, 1), dtype=tf.int32),
    keras.layers.Conv2D(4, kernel_size=(2, 2), strides=2, activation="relu"),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Conv2D(8, kernel_size=(2, 2), strides=1, activation="relu"),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(500),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(100),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(50),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(num_actions),
])

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

train_step_counter = tf.Variable(0)

agent = dqn_agent.DqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=train_step_counter)

agent.initialize()

eval_policy = agent.policy
collect_policy = agent.collect_policy

random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                train_env.action_spec())

example_environment = tf_py_environment.TFPyEnvironment(new_env())

time_step = example_environment.reset()

random_policy.action(time_step)

def compute_avg_return(environment, policy, num_episodes=10):

  total_return = 0.0
  for _ in range(num_episodes):

    time_step = environment.reset()
    episode_return = 0.0

    while not time_step.is_last():
      action_step = policy.action(time_step)
      time_step = environment.step(action_step.action)
      episode_return += time_step.reward
    total_return += episode_return

  avg_return = total_return / num_episodes
  return avg_return.numpy()[0]


compute_avg_return(eval_env, random_policy, num_eval_episodes)

table_name = 'uniform_table'
replay_buffer_signature = tensor_spec.from_spec(
      agent.collect_data_spec)
replay_buffer_signature = tensor_spec.add_outer_dim(
    replay_buffer_signature)

table = reverb.Table(
    table_name,
    max_size=replay_buffer_max_length,
    sampler=reverb.selectors.Uniform(),
    remover=reverb.selectors.Fifo(),
    rate_limiter=reverb.rate_limiters.MinSize(1),
    signature=replay_buffer_signature)

reverb_server = reverb.Server([table])

replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
    agent.collect_data_spec,
    table_name=table_name,
    sequence_length=2,
    local_server=reverb_server)

print('replay_buffer')

rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
  replay_buffer.py_client,
  table_name,
  sequence_length=2)

print('rb_observer')

agent.collect_data_spec

agent.collect_data_spec._fields

py_driver.PyDriver(
    env,
    py_tf_eager_policy.PyTFEagerPolicy(
      random_policy, use_tf_function=True),
    [rb_observer],
    max_steps=initial_collect_steps).run(train_py_env.reset())

# For the curious:
# Uncomment to peel one of these off and inspect it.
# iter(replay_buffer.as_dataset()).next()

# Dataset generates trajectories with shape [Bx2x...]
dataset = replay_buffer.as_dataset(
    num_parallel_calls=3,
    sample_batch_size=batch_size,
    num_steps=2).prefetch(3)
experience_dataset_fn = lambda: dataset

print('dataset')

iterator = iter(dataset)
print(iterator)

agent.train = common.function(agent.train)

saved_model_dir = os.path.join(data_dir, learner.POLICY_SAVED_MODEL_DIR)

learning_triggers = [
    triggers.PolicySavedModelTrigger(
        saved_model_dir,
        agent,
        train_step_counter,
        interval=policy_save_interval),
    triggers.StepPerSecondLogTrigger(train_step_counter, interval=1000),
]

agent_learner = learner.Learner(
    data_dir,
    train_step_counter,
    agent,
    experience_dataset_fn,
    triggers=learning_triggers,
    checkpoint_interval=policy_save_interval)

step = agent_learner.train_step_numpy
print("Loaded at step", step)

print('before compute_avg_return')
avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
print('after compute_avg_return')
returns = [avg_return]

time_step = train_py_env.reset()

collect_driver = py_driver.PyDriver(
    env,
    py_tf_eager_policy.PyTFEagerPolicy(
      agent.collect_policy, use_tf_function=True),
    [rb_observer],
    max_steps=collect_steps_per_iteration)

print('collect_driver')

for _ in range(num_iterations):
  time_step, _ = collect_driver.run(time_step)

  #experience, unused_info = next(iterator)
  #train_loss = agent.train(experience).loss

  #step = agent.train_step_counter.numpy()

  loss_info = agent_learner.run(iterations=1, iterator=iterator)
  step = agent_learner.train_step_numpy

  if step % log_interval == 0:
    print('step = {0}: loss = {1}'.format(step, loss_info.loss.numpy()))

  if step % eval_interval == 0:
    avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
    print('step = {0}: Average Return = {1}'.format(step, avg_return))
    returns.append(avg_return)

iterations = range(0, num_iterations + 1, eval_interval)
#plt.plot(iterations, returns)
#plt.ylabel('Average Return')
#plt.xlabel('Iterations')
#plt.ylim(top=250)
#plt.show()

def create_policy_eval_video(policy, filename, num_episodes=5, fps=10):
  filename = filename + ".mp4"
  with imageio.get_writer(filename, fps=fps) as video:
    for _ in range(num_episodes):
      time_step = eval_env.reset()
      video.append_data(eval_py_env.render())
      while not time_step.is_last():
        action_step = policy.action(time_step)
        time_step = eval_env.step(action_step.action)
        video.append_data(eval_py_env.render())

create_policy_eval_video(random_policy, "custom-game-random")
create_policy_eval_video(agent.policy, f"custom-game-trained-{step}")
