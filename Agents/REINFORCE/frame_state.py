from collections import deque

class frame_state(object):
    def __init__(self, k):
        self.k = k
        self.frames = deque([], maxlen=k)

    def reset(self, state):
        for _ in range(self.k):
            self.frames.append(state)
        #return self._get_ob()

    def step(self, state):
        self.frames.append(state)
        return self._get_ob()

    def _get_ob(self):
        assert len(self.frames) == self.k
        return list(self.frames)
