import pyglet
import time
import math
import random
from typing import Optional

import numpy as np
from collections import Counter
from words import word_list

def get_display(spec):
    """Convert a display specification (such as :0) into an actual Display
    object.
    Pyglet only supports multiple Displays on Linux.
    """
    if spec is None:
        return pyglet.canvas.get_display()
        # returns already available pyglet_display,
        # if there is no pyglet display available then it creates one
    elif isinstance(spec, str):
        return pyglet.canvas.Display(spec)
    else:
        raise error.Error(
            f"Invalid display specification: {spec}. (Must be a string like :0 or None.)"
        )


def get_window(width, height, display, **kwargs):
    """
    Will create a pyglet window from the display specification provided.
    """
    screen = display.get_screens()  # available screens
    config = screen[0].get_best_config()  # selecting the first screen
    context = config.create_context(None)  # create GL context

    return pyglet.window.Window(
        width=width,
        height=height,
        display=display,
        config=config,
        context=context,
        **kwargs,
    )

def make_observation(target, guess):
    letter_counts = dict(Counter(target))

    observation = [1] * 5 # all null
    guess_letters_used = {c: 0 for c in guess}

    # fill in the exact hits
    for i, (cg,tg) in enumerate(zip(guess, target)):
        if cg == tg:
            observation[i] = 3 
            guess_letters_used[cg] += 1

    # fill in the wrong position hits
    for i, (cg,tg) in enumerate(zip(guess, target)):
        if cg in letter_counts and observation[i] == 0:
            if guess_letters_used[cg] < letter_counts[cg]:
                observation[i] = 2
                guess_letters_used[cg] += 1

    return observation

class WordleEnv:
    def __init__(self):
        self.viewer = None
        self.n_actions = len(word_list)
        self.guessed_words = []
        self.letter_colors = []
        self.color_lookup = {
            0: (0, 0, 0, 0), 
            1: (255, 0, 0, 255), 
            2: (0, 0, 255, 255),
            3: (0, 255, 0, 255),
        }
        
        self.actions = 6 * [0] # 6 words
        self.scores = 6  * [5 * [0]] # 6 words * 5 letters
        self.window = None

    def step(self, action):
        # get the word from the action
        guess = word_list[action]
        score = make_observation(self.secret_word, guess)
        self.guessed_words.append(guess)
        self.letter_colors.append([self.color_lookup[i] for i in score])
        
        reward = 1 if all([s == 3 for s in score]) else -1
        
        # update the state
        done = False
        if len(self.guessed_words) == 6 or reward == 1:
            done = True
        
        #state = self.render(return_image=True)        
        n_guesses = len(self.guessed_words)
        self.actions[n_guesses - 1] = action
        self.scores[n_guesses - 1] = score
        state = np.concatenate([self.actions, *self.scores], axis=0) 

        return state, reward, done, {}

    def reset(self, seed: Optional[int] = None):
        # set a random word
        if self.window is not None:
            self.window.clear()
        self.guessed_words = []
        self.letter_colors = []
        self.actions = 6 * [0] # 6 words
        self.scores = 6  * [5 * [0]] # 6 words * 5 letters
        state = np.concatenate([self.actions, *self.scores], axis=0) 

        self.secret_word = random.choice(word_list)
        return state

    def render(self, return_image=False):
        if self.window is None:
            display = get_display(None)
            self.screen_width = 64
            self.screen_height = 96
            self.window = get_window(self.screen_width, self.screen_height, display)

        labels = []
        for i,word in enumerate(self.guessed_words):
            y = (6 - i) * (self.screen_height // 6) - 6
            for c,(letter,color) in enumerate(zip(word, self.letter_colors[i])):
                x = (c) * (self.screen_width // 5) + 8
                text = pyglet.text.Label(
                    letter, font_name='Monospace',
                    font_size=10, x=x, y=y,
                    anchor_x='center', anchor_y='center',
                    color=color
                )
                labels.append(text)
        
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        for label in labels:
            label.draw()

        self.window.flip()
        time.sleep(1)

        if return_image:
            buffer = pyglet.image.get_buffer_manager().get_color_buffer()
            image_data = buffer.get_image_data()
            arr = np.frombuffer(image_data.get_data(), dtype=np.uint8)
            arr = arr.reshape(buffer.height, buffer.width, 4)
            return arr[..., :3] # leave out alpha
        

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
