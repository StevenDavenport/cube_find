from minerl.herobraine.env_specs.simple_embodiment import SimpleEmbodimentEnvSpec
from minerl.herobraine.hero.handler import Handler
import minerl.herobraine.hero.handlers as handlers
from minerl.env._singleagent import _SingleAgentEnv
from typing import List
import random
import clip
import torch
from PIL import Image

CubeFind_DOC = """ This environment is a simple embodiment environment
where the agent is tasked with finding a cube in a room. """


class CubeFindEnv(SimpleEmbodimentEnvSpec, _SingleAgentEnv):
    def __init__(self, level : str = 1, find_length : int = 8000, number_of_cubes : int = 20, reward_threshold : int = 100, *args, **kwargs):
        self.level = level
        self.cube_find_length = find_length
        self.number_of_cubes = number_of_cubes
        self.reward_threshold = reward_threshold
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.goal = clip.tokenize(["A blue cube next to a red cube"]).to(self.device)
        if 'name' not in kwargs:
            kwargs['name'] = 'CubeFind-v0'
        self.cube_list = []
        super().__init__(*args, max_episode_steps=self.cube_find_length, reward_threshold=self.reward_threshold, **kwargs)

    def create_server_world_generators(self) -> List[Handler]:
        return [
            handlers.FlatWorldGenerator(generatorString="1;7,2x3,2;1"),
            handlers.DrawingDecorator(
                "<DrawBlock x=\"0\" y=\"4\" z=\"20\" type=\"wool\" colour=\"BLUE\"/>"
                "<DrawBlock x=\"1\" y=\"4\" z=\"20\" type=\"wool\" colour=\"RED\"/>"
                "<DrawBlock x=\"20\" y=\"4\" z=\"0\" type=\"wool\" colour=\"GREEN\"/>"
                "<DrawBlock x=\"20\" y=\"4\" z=\"1\" type=\"wool\" colour=\"YELLOW\"/>"
                "<DrawBlock x=\"0\" y=\"4\" z=\"-20\" type=\"wool\" colour=\"PURPLE\"/>"
                "<DrawBlock x=\"1\" y=\"4\" z=\"-20\" type=\"wool\" colour=\"ORANGE\"/>"
                "<DrawBlock x=\"-20\" y=\"4\" z=\"0\" type=\"wool\" colour=\"WHITE\"/>"
                "<DrawBlock x=\"-20\" y=\"4\" z=\"1\" type=\"wool\" colour=\"BLACK\"/>"
                "<DrawCuboid x1=\"-200\" y1=\"3\" z1=\"25\" x2=\"200\" y2=\"-10\" z2=\"225\" type=\"air\"/>"
                "<DrawCuboid x1=\"-200\" y1=\"3\" z1=\"-25\" x2=\"200\" y2=\"-10\" z2=\"-225\" type=\"air\"/>"
                "<DrawCuboid x1=\"-200\" y1=\"3\" z1=\"-200\" x2=\"-25\" y2=\"-10\" z2=\"200\" type=\"air\"/>"
                "<DrawCuboid x1=\"200\" y1=\"3\" z1=\"-200\" x2=\"25\" y2=\"-10\" z2=\"200\" type=\"air\"/>"
            )
        ]

    def step(self, action):
        obs, reward, done, info = super().step(action)
        return obs, self.calculate_reward(obs), done, info

    def calculate_reward(self, obs):
        pov = self.preprocess(obs['pov']).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.clip.encode_image(pov)
            text_features = self.clip.encode_text(self.goal)
            logits_per_image, logits_per_text = self.clip(image_features, text_features)
            #probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        return logits_per_image

    # not using this now but keeping it as it will eventually be useful 
    def cube_generator(self) -> str:
        xml = ""
        available_colours = ["RED", "BLUE", "GREEN", "YELLOW", "PURPLE", "ORANGE", "WHITE", "BLACK"]
        for i in range(self.number_of_cubes):
            x = random.randint(-10, 10)
            z = random.randint(-10, 10)
            xml += f"<DrawBlock x=\"{x}\" y=\"5\" z=\"{z}\" type=\"wool\" colour=\"{'YELLOW'}\"/>" if i == 0 else \
            f"<DrawBlock x=\"{x}\" y=\"5\" z=\"{z}\" type=\"wool\" colour=\"{random.choice(available_colours)}\"/>"
            #self.level >=2 f<DrawBlock""
            if self.level >= 2:
                xml += f"<DrawBlock x=\"{x}\" y=\"6\" z=\"{z}\" type=\"wool\" colour=\"{'BLUE'}\"/>" if i == 0 else \
                f"<DrawBlock x=\"{x}\" y=\"6\" z=\"{z}\" type=\"wool\" colour=\"{random.choice(available_colours)}\"/>"
            #colours_to_choose_from.remove(colour)
        return xml

    def create_agent_start(self) -> List[Handler]:
        self.create_server_initial_conditions()
        return [
            # make the agent start with these items
            handlers.SimpleInventoryAgentStart([
                dict(type="water_bucket", quantity=1),
                dict(type="diamond_pickaxe", quantity=1)
            ]),
            # Start position
            handlers.AgentStartPlacement(0, 5, 0, 0, 0)
            # set to the morning
        ]

    def create_rewardables(self) -> List[Handler]:
        return [] # make a reward, if pos found CLIP


    def create_agent_handlers(self) -> List[Handler]:
        return [
            # make the agent quit when it gets a gold block in its inventory
            handlers.AgentQuitFromPossessingItem([
                dict(type="gold_block", amount=1)
            ])
        ]

    def create_actionables(self) -> List[Handler]:
        return super().create_actionables() + [
            # allow agent to place water
            handlers.KeybasedCommandAction("use"),
            # also allow it to equip the pickaxe
            handlers.EquipAction(["diamond_pickaxe"])
        ]
        
    def create_observables(self) -> List[Handler]:
        return super().create_observables() + [
            # current location and lifestats are returned as additional
            # observations
            handlers.ObservationFromCurrentLocation(),
            handlers.ObservationFromLifeStats()
        ]

    def create_server_initial_conditions(self) -> List[Handler]:
        return [
            # Sets time to morning and stops passing of time
            handlers.TimeInitialCondition(False, 8000)
        ]

    # see API reference for use cases of these first two functions

    def create_server_quit_producers(self):
        return []

    def create_server_decorators(self) -> List[Handler]:
        return []

    # the episode can terminate when this is True
    def determine_success_from_rewards(self, rewards: list) -> bool:
        return sum(rewards) >= self.reward_threshold

    def is_from_folder(self, folder: str) -> bool:
        return folder == 'CubeFind'

    def get_docstring(self):
        return CubeFind_DOC

if __name__ == "__main__":
    import gym

    abs_CubeFind = CubeFindEnv()
    abs_CubeFind.register()
    env = gym.make("CubeFind-v0")
    obs  = env.reset()

    done = False
    while not done:
        env.render()
        action = env.action_space.noop()
        action['camera'][1] = +0.25
        #action['forward'] = 1
        obs, reward, done, info = env.step(action)

