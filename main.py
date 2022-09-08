from re import I
from envs.cube_find import CubeFindEnv
import minerl
import gym

def main():
    abs_CubeFindEnv = CubeFindEnv()
    abs_CubeFindEnv.register()
    env = gym.make('CubeFind-v0')
    obs = env.reset()

    #dunno what this does, test it
    #env = gym.wrappers.Monitor(env, "recording", force=True)

    done = False
    while not done:
        env.render()

        action = env.action_space.noop() # wrap this?
        action['camera'][1] = +0.25
        obs, reward, done, info = env.step(action) # obs?


if __name__ == '__main__':
    main()