import random
import pygame

from studente import Node, successors, heuristic, a_star


def pathfind(start, goal, is_solid, region_width, region_height):
    sol = a_star(
        initial_state=start,
        goal_test=lambda s: s == goal,
        successor_fn=lambda s: successors(s, is_solid=is_solid, region_width=region_width, region_height=region_height),
        heuristic_fn=lambda s: heuristic(s, goal=goal, is_solid=is_solid),
    )
    if sol is None:
        return None
    else:
        return sol.to_solution()



# Environment implementation

class World:

    def __init__(self, region_width, region_height, pathfinder=None):
        self.region_width = region_width
        self.region_height = region_height
        self.pathfinder = pathfinder
        self.start = (0, 0)
        self.pos = self.start
        self.solid_map = {(0, 0): False}
        self.new_random_goal()
    
    def new_random_goal(self):
        goal = None
        while goal is None:
            goal = (
                random.randrange(0, self.region_width),
                random.randrange(0, self.region_height)
            )
            if hasattr(self, 'goal') and goal == self.goal:
                goal = None
            if goal == self.pos:
                goal = None
            if self.is_solid(goal):
                goal = None
        self.goal = goal
        self.start = self.pos
        print(f'\nNew goal: {self.goal} (start: {self.pos})')
        if self.pathfinder is not None:
            path = pathfind(
                start=self.start,
                goal=self.goal,
                is_solid=self.is_solid,
                region_width=self.region_width,
                region_height=self.region_height
            )
            if path is None:
                print(f'No path found from {self.start} to {self.goal}, giving up current goal ({self.goal}).')
                self.new_random_goal()
                return
            print(f'Found path from {self.start} to {self.goal}: {path}')
            
            self.inv_path = list(reversed(path))

    def is_solid(self, pos):
        if pos not in self.solid_map:
            self.solid_map[pos] = random.choice([True, True, True, False, False, False, False])
        return self.solid_map[pos]

    def is_inside_region(self, pos):
        return pos[0] >= 0 and pos[0] < self.region_width and pos[1] >= 0 and pos[1] < self.region_height

    def tick_interactive(self):
        if not hasattr(self, 'pressed'):
            self.pressed = pygame.key.get_pressed()
        delta, self.pressed = get_action_from_keyboard(self.pressed)
        new_pos = (self.pos[0] + delta[0], self.pos[1] + delta[1])
        if self.is_solid(new_pos):
            new_pos = self.pos
        if not self.is_inside_region(new_pos):
            new_pos = self.pos
        self.pos = new_pos
        if self.pos == self.goal:
            print(f'Goal reached!')
            self.new_random_goal()

    def tick_ai(self):
        if self.pos == self.goal:
            print(f'Goal {self.goal} reached!')
            self.new_random_goal()
        if len(self.inv_path) == 0:
            print(f'Path ended before goal {self.goal} reached, giving up current goal ({self.goal}).')
            self.new_random_goal()
        new_pos = self.inv_path.pop()
        if self.is_solid(new_pos):
            print(f'Trying to go to a solid cell {new_pos}, giving up current goal ({self.goal}).')
            self.new_random_goal()
        elif not self.is_inside_region(new_pos):
            print(f'Trying to go outside the region, giving up current goal ({self.goal}).')
            self.new_random_goal()
        elif abs(self.pos[0] - new_pos[0]) + abs(self.pos[1] - new_pos[1]) >= 2:
            print(f'Trying an invalid move: {self.pos} to {new_pos}, giving up current goal ({self.goal}).')
            self.new_random_goal()
        else:
            self.pos = new_pos        

    def tick(self):
        if self.pathfinder is None:
            self.tick_interactive()
        else:
            self.tick_ai()

    def render(self, screen):
        w, h = screen.get_width(), screen.get_height()
        screen.fill("cadetblue4")

        # draw region bg color
        w_r, h_r = self.region_width, self.region_height
        # origin in region coords
        xo, yo = 1, 1
        # scale (region to cell coords)
        s = int(min(w/(w_r+2*xo), h/(h_r+2*yo)))
        pygame.draw.rect(
            screen,
            color="white",
            rect=pygame.Rect(
                    s*xo,
                    s*yo,
                    s*w_r,
                    s*h_r,
            )
        )

        # draw solid blocks
        for x in range(-xo, int(w/s)+1):
            for y in range(-yo, int(h/s)+1):
                if not self.is_solid((x, y)) or x >= w_r or x < 0 or y < 0 or y >= h_r:
                    continue
                pygame.draw.rect(
                    screen,
                    color="darkgray",
                    rect=pygame.Rect(s*(xo+x), s*(yo+y), s, s)
                )

        # draw start, goal
        pygame.draw.rect(
            screen,
            color="lightblue",
            rect=pygame.Rect(s*(xo+self.start[0]), s*(yo+self.start[1]), s, s)
        )
        pygame.draw.rect(
            screen,
            color="green",
            rect=pygame.Rect(s*(xo+self.goal[0]), s*(yo+self.goal[1]), s, s)
        )

        # draw grid
        # horizontal lines
        for y in [ s*y_r for y_r in range(0, int(h/s)+1) ]:
            pygame.draw.line(
                screen,
                color="black",
                start_pos=(0, y),
                end_pos=(w, y),
            )
        # vertical lines
        for x in [ s*x_r for x_r in range(0, int(w/s)+1) ]:
            pygame.draw.line(
                screen,
                color="black",
                start_pos=(x, 0),
                end_pos=(x, h),
            )

        # TODO draw agent
        pygame.draw.circle(
            screen,
            color="blue",
            center=(s/2+s*(xo+self.pos[0]), s/2+s*(yo+self.pos[1])),
            radius=(7/10)*s/2
        )


def get_action_from_keyboard(last_pressed):
    keyboard = pygame.key.get_pressed()
    dx, dy = 0, 0
    if keyboard[pygame.K_LEFT] and not last_pressed[pygame.K_LEFT]:
        dx = -1
    if keyboard[pygame.K_RIGHT] and not last_pressed[pygame.K_RIGHT]:
        dx = +1
    if keyboard[pygame.K_UP] and not last_pressed[pygame.K_UP]:
        dy = -1
    if keyboard[pygame.K_DOWN] and not last_pressed[pygame.K_DOWN]:
        dy = +1
    return (dx, dy), keyboard



def run(env):
    # pygame setup
    pygame.init()
    screen = pygame.display.set_mode((1280, 720))
    clock = pygame.time.Clock()
    running = True
    frames = 1
    frames_per_step = 1 if env.pathfinder is None else 20
    env.render(screen=screen)

    while running:
        # poll for events
        # pygame.QUIT event means the user clicked X to close your window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # advance world by one time step and render it to screen
        if frames % frames_per_step == 0:
            env.tick()
            env.render(screen=screen)
        frames += 1

        # flip() the display to put your work on screen
        pygame.display.flip()

        clock.tick(60)  # limits FPS to 60

    print(f'Closing...')

    pygame.quit()


if __name__ == '__main__':
    random.seed(9438, version=2)
    env = World(region_width=16, region_height=9, pathfinder=pathfind)
    run(env)